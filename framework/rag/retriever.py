"""RAG retriever — chunk + embed texts, expose retrieval as an agentic toolset.

One-shot retrieval (`.as_tool()`) does a single fixed top-k search: one query,
one shot, no recovery. `.as_tools()` upgrades that to a *navigable* set the agent
loops over, which is what turns retrieval from a pre-processing step into a
reasoning surface:

  - search_documents(query, k) — semantic top-k, but it prints an [idx N] for every
    hit so the result is *addressable*. That addressability is what makes multi-hop
    chaining and beyond-k accumulation possible.
  - lookup_term(term)          — a LEXICAL scan (exact substring), the fallback for
    when semantic search misses on vocabulary mismatch. Semantic and lexical fail on
    different inputs; the agent picks.
  - get_section(index, window) — expand a hit into its neighbouring chunks, so the
    agent can pull in evidence the k-limit structurally excluded.

Default embeddings are Google's (needs GOOGLE_API_KEY); inject any Embeddings for
tests or other providers.
"""

import os

from langchain_core.tools import create_retriever_tool, tool
from langchain_core.vectorstores import InMemoryVectorStore

from framework.rag.chunker import chunk_text

# Caps keep a tool result from becoming the context bloat we're trying to avoid.
MAX_LEXICAL_HITS = 8


class Retriever:
    def __init__(self, texts, *, embeddings=None, k=3, chunk_size=300, overlap=50):
        self._embeddings = embeddings or self._default_embeddings()
        self._k = k
        docs = self._load(texts)
        # Keep an ordered, indexed chunk list — chunk identity is what get_section
        # navigates by, and what search/lookup surface as [idx N].
        self._chunks: list[str] = []
        for d in docs:
            self._chunks.extend(chunk_text(d, chunk_size, overlap))
        metadatas = [{"idx": i} for i in range(len(self._chunks))]
        self._store = InMemoryVectorStore.from_texts(
            self._chunks, embedding=self._embeddings, metadatas=metadatas,
        )

    @staticmethod
    def _default_embeddings():
        # Global default = bge-m3 (local, cached); Gemini is the secondary option.
        # Swap via EMBED_PROVIDER env or Retriever(embeddings=make_embeddings("gemini")).
        from framework.rag.embeddings import make_embeddings
        return make_embeddings()

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

    # --- formatting helpers -------------------------------------------------

    @staticmethod
    def _fmt(idx: int, text: str) -> str:
        return f"[idx {idx}] {text}"

    # --- the toolset --------------------------------------------------------

    def as_tools(self) -> list:
        """Five... well, three-and-growing navigable retrieval tools (Path A).

        Structure-dependent tools (list_outline / get_table) are deferred until the
        corpus carries structure; these three work on any flat chunk set.
        """
        store = self._store
        chunks = self._chunks
        default_k = self._k
        fmt = self._fmt

        @tool
        def search_documents(query: str, k: int = 0) -> str:
            """Semantically search the knowledge base for passages relevant to `query`.

            Returns the top matches, each prefixed with an [idx N] address. Use a hit's
            idx with get_section to read the surrounding context. Raise `k` to widen the
            net; re-run with reworded queries if the first pass misses."""
            n = k or default_k
            hits = store.similarity_search(query, k=n)
            if not hits:
                return f"No passages matched '{query}'. Try rewording or lookup_term."
            return "\n".join(fmt(h.metadata["idx"], h.page_content) for h in hits)

        @tool
        def lookup_term(term: str) -> str:
            """Find chunks containing the EXACT text `term` (case-insensitive substring).

            This is a lexical fallback for when semantic search misses — a name, code,
            or acronym you saw in one chunk and want to trace elsewhere. Complements
            search_documents; they fail on different inputs."""
            needle = term.lower()
            hits = [(i, c) for i, c in enumerate(chunks) if needle in c.lower()]
            if not hits:
                return f"No chunks contain '{term}'."
            shown = hits[:MAX_LEXICAL_HITS]
            body = "\n".join(fmt(i, c) for i, c in shown)
            if len(hits) > MAX_LEXICAL_HITS:
                body += f"\n... ({len(hits) - MAX_LEXICAL_HITS} more; narrow the term)"
            return body

        @tool
        def get_section(index: int, window: int = 1) -> str:
            """Read chunk `index` plus `window` neighbours on each side, in order.

            Use it to expand a search hit into its surrounding context — the way past
            the top-k ceiling when an answer needs evidence the search didn't rank."""
            if not chunks:
                return "The knowledge base is empty."
            if index < 0 or index >= len(chunks):
                return (f"No chunk at idx {index} (valid range 0..{len(chunks) - 1}). "
                        "Use search_documents or lookup_term to find a valid idx.")
            lo = max(0, index - window)
            hi = min(len(chunks), index + window + 1)
            return "\n".join(fmt(i, chunks[i]) for i in range(lo, hi))

        return [search_documents, lookup_term, get_section]

    def as_tool(self, name: str = "search_documents",
                description: str = "Search the knowledge base for relevant passages."):
        """One-shot alias, kept for backward compatibility with existing callers."""
        retriever = self._store.as_retriever(search_kwargs={"k": self._k})
        return create_retriever_tool(retriever, name, description)


class DocumentRetriever:
    """Agentic retrieval over a fully-parsed `ParsedDoc` — the Phase-5 convergence of
    the ingestion pipeline (parse → chunk → enrich → hybrid+rerank) with the navigable
    toolset. Where `Retriever.as_tools()` gave 3 structure-blind tools, this exposes
    five STRUCTURE-AWARE tools so an agent can navigate, not just search:

      - search_documents  hybrid (dense+BM25+RRF) + optional cross-encoder rerank
      - list_outline      the document's table of contents (orient before searching)
      - get_section       a whole section's text by title (follow a cross-reference)
      - lookup_term       a real glossary hit (no retrieval round-trip)
      - get_table         a table verbatim (never flattened)

    `scorer` (a reranker, e.g. rerank.default_scorer()) is optional; without it search
    is hybrid-only. Embeddings default to the framework default (bge-m3, cached).
    """

    def __init__(self, doc, *, embeddings=None, scorer=None, k: int = 5, candidates: int = 50):
        from framework.rag.chunker import structure_aware_chunks
        from framework.rag.enrich import enrich_chunks
        from framework.rag.embeddings import make_embeddings
        from framework.rag.index import HybridIndex

        self._doc = doc
        self._chunks = enrich_chunks(structure_aware_chunks(doc), glossary=doc.glossary)
        self._index = HybridIndex(self._chunks, embeddings or make_embeddings())
        self._scorer = scorer
        self._k = k
        self._candidates = candidates

    def _search(self, query: str, k: int) -> list:
        hits = self._index.retrieve(query, self._candidates)
        if self._scorer is not None:
            from framework.rag.rerank import rerank
            return rerank(query, hits, k, scorer=self._scorer)
        return hits[:k]

    def as_tools(self) -> list:
        doc, chunks, k = self._doc, self._chunks, self._k
        search = self._search

        @tool
        def search_documents(query: str) -> str:
            """Search the knowledge base for passages relevant to `query` (hybrid dense +
            keyword, reranked). The best first move for most questions; rerun with reworded
            queries if the first pass misses."""
            hits = search(query, k)
            if not hits:
                return f"No passages matched '{query}'."
            return "\n\n".join(
                f"[{' > '.join(h['section_path']) or 'doc'} | p{h['page']}]\n{h['text']}"
                for h in hits)

        @tool
        def list_outline() -> str:
            """Return the document's table of contents (section titles + pages). Use it to
            orient before searching, or to find a section name to pass to get_section."""
            if not doc.outline:
                return "No outline available."
            return "\n".join(f"{'  ' * (o.level - 1)}- {o.title} (p{o.page})"
                             for o in doc.outline)

        @tool
        def get_section(section_title: str) -> str:
            """Return the full text of a section by (partial, case-insensitive) title —
            for following an explicit cross-reference like 'see the section on X'."""
            t = section_title.lower()
            hits = [c for c in chunks if any(t in s.lower() for s in c.section_path)]
            if not hits:
                return f"No section matching '{section_title}'. Try list_outline."
            seen, out = set(), []
            for c in hits:
                if c.text not in seen:
                    seen.add(c.text)
                    out.append(c.text)
            return "\n\n".join(out)[:4000]

        @tool
        def lookup_term(term: str) -> str:
            """Look up an abbreviation or term in the document's glossary — a direct hit,
            no retrieval round-trip. Use for acronyms and defined terms."""
            entry = doc.glossary.get(term) or doc.glossary.get(term.upper())
            if not entry:
                return f"'{term}' is not in the glossary."
            out = f"{term.upper()}: {entry.expansion}"
            if entry.definition:
                out += f" — {entry.definition}"
            return out

        @tool
        def get_table(query: str) -> str:
            """Fetch a table verbatim (Markdown) matching a keyword or section name.
            Tables hold data prose can't; read them whole rather than via search."""
            t = query.lower()
            tables = [c for c in chunks if c.kind == "table"
                      and (t in c.text.lower() or any(t in s.lower() for s in c.section_path))]
            if not tables:
                return f"No table matching '{query}'."
            return tables[0].text

        return [search_documents, list_outline, get_section, lookup_term, get_table]
