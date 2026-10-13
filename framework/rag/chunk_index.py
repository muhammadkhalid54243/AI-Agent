"""Minimal chunk index — the Phase-2 MEASUREMENT surface only.

Indexes the leaf units (children + tables) of a structure-aware chunking over their
`embed_text`, carrying page/section metadata so retrieval hits are citable and
`score_retrieval` can match on section as well as phrase. The real hybrid dense+BM25
index with reranking and child->parent expansion is Phase 4 — this is deliberately
small, just enough to score chunking quality against the gold sets.
"""

from langchain_core.vectorstores import InMemoryVectorStore

from framework.rag.chunker import leaf_chunks


class ChunkIndex:
    def __init__(self, chunks: list, embeddings):
        leaves = leaf_chunks(chunks)
        texts = [c.embed_text for c in leaves]
        metadatas = [{"chunk_id": c.id, "page_start": c.page_start, "page_end": c.page_end,
                      "printed_page": c.printed_page_start,
                      "section_path": list(c.section_path), "kind": c.kind, "text": c.text}
                     for c in leaves]
        self.n_indexed = len(leaves)
        self._store = InMemoryVectorStore.from_texts(texts, embedding=embeddings,
                                                     metadatas=metadatas)

    def retrieve(self, query: str, k: int) -> list:
        # with_score so the raw cosine survives to the confidence gate (InMemoryVectorStore
        # returns cosine similarity, higher = more similar). RRF downstream is rank-only and
        # cannot be thresholded, so this is the one absolute relevance signal we keep.
        pairs = self._store.similarity_search_with_score(query, k=k)
        return [{"text": h.metadata.get("text", h.page_content),
                 "page": h.metadata.get("page_start"),
                 "printed_page": h.metadata.get("printed_page"),
                 "section_path": h.metadata.get("section_path"),
                 "chunk_id": h.metadata.get("chunk_id"),
                 "kind": h.metadata.get("kind"),
                 "dense_score": score} for h, score in pairs]
