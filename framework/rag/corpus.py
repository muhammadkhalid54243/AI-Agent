"""Corpus retrieval — one hybrid index over MANY documents.

`DocumentRetriever` indexes a single `ParsedDoc`. `CorpusRetriever` is the multi-doc
counterpart the serve layer uses: it parses/enriches each doc on its own (breadcrumb
and glossary enrichment are doc-specific), then concatenates the chunks into ONE
`HybridIndex`. This works unchanged because chunk ids are already namespaced by doc
(`{doc_id}:ck{n}` in chunker.py), so ids never collide and every hit can name the
report it came from via `doc_id`.

Why merged rather than per-doc + a router: it ranks across the whole corpus in one
shot, cross-document questions fall out for free, and BM25's IDF is computed over the
real corpus. A router is only worth its misroute risk once measurement shows the flat
index actually dilutes recall — see examples/multidoc_eval.py.
"""

import re


def _label(doc) -> str:
    """A short, human citation label for a document.

    `2024_SHR_English` -> `SHR 2024`; other docs -> the doc_id with separators
    normalised. Citations read like `[SHR 2024 > Key issues | p12]`.
    """
    did = doc.doc_id
    m = re.search(r"(19|20)\d{2}", did)
    year = m.group(0) if m else ""
    if "SHR" in did.upper():
        return f"SHR {year}".strip()
    return re.sub(r"[_-]+", " ", did).strip()


class CorpusRetriever:
    def __init__(self, docs, *, embeddings=None, scorer=None, k: int = 5,
                 candidates: int = 50):
        from framework.rag.chunker import structure_aware_chunks
        from framework.rag.enrich import enrich_chunks
        from framework.rag.embeddings import make_embeddings
        from framework.rag.index import HybridIndex

        docs = list(docs)
        if not docs:
            raise ValueError("CorpusRetriever needs at least one document")
        self._docs = {d.doc_id: d for d in docs}

        # Enrich per-doc (breadcrumb + glossary are doc-specific), then merge. Prepend the
        # document label (e.g. "[SHR 2024]") into each chunk's EMBED_TEXT only — never the
        # returned text, so citations stay clean. Near-duplicate annual reports are almost
        # identical passage-for-passage; without this tag a 2023 chunk outranks the correct
        # 2024 one at rank 1 (measured: doc_attribution 1.0 -> 0.656 when 2023 is added).
        # The tag separates them in both channels: dense (the year shifts the vector) and
        # BM25 (the year/report becomes a matchable token).
        chunks = []
        for d in docs:
            label = _label(d)
            for c in enrich_chunks(structure_aware_chunks(d), glossary=d.glossary):
                c.embed_text = f"[{label}] {c.embed_text}"
                chunks.append(c)
        self._chunks = chunks
        self._index = HybridIndex(chunks, embeddings or make_embeddings())

        # Merged glossary for acronym notes; first doc wins on a clashing expansion.
        self.glossary = {}
        for d in docs:
            for term, entry in d.glossary.items():
                self.glossary.setdefault(term, entry)

        self._labels = {d.doc_id: _label(d) for d in docs}
        self._scorer = scorer
        self._k = k
        self._candidates = candidates

    def doc_label(self, doc_id: str) -> str:
        return self._labels.get(doc_id, doc_id)

    def _search(self, query: str, k: int) -> list:
        hits = self._index.retrieve(query, self._candidates)
        if self._scorer is not None:
            from framework.rag.rerank import rerank
            return rerank(query, hits, k, scorer=self._scorer)
        return hits[:k]
