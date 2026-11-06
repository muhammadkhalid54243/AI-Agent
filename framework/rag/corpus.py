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

NOTE: near-duplicate collapse (Phase E) was attempted and REVERTED — it measured a net
regression on the current corpus because collapse fights the inconsistent cross-year
breadcrumbs (`get_section` locates by breadcrumb title, which the collapsed representative
no longer matches). Parked until the breadcrumb-nesting bug is fixed.
"""

import re

# Enumerate/table dispatch returns a WHOLE section/table, capped so a huge chapter can't
# blow the context window — well above RAG_K, but bounded (Phase D).
SECTION_TOKEN_BUDGET = 3000

# Deterministic abstention-gate thresholds. The serve layer abstains (returns NOT_COVERED,
# no model call) when retrieval confidence falls below these — moving the answer-vs-abstain
# decision OUT of the stochastic model, which flip-flopped on identical context. CALIBRATED in
# examples/abstention_eval.py against 65 gold positives + 12 hard negatives (full English corpus).
#
# MEASURED (dense, override off): positives cosine [0.489, 0.781] median 0.630; negatives
# [0.341, 0.617]. The bands OVERLAP — bge-m3 embeds TOPIC, so topical near-misses ("death
# sentences in India") score as high as real questions and dense cosine cannot reject them.
# 0.45 answers all 65 positives (false-abstain 0) and rejects only clearly-out-of-domain
# negatives (France, cookies, boiling point, World Cup, space policy). Rejecting topical
# near-misses is Phase G's verifier job, by design — not a threshold we can set here.
MIN_DENSE = 0.45        # bge-m3 cosine of the best retrieved chunk (reranker off — the default)
# bge-reranker-v2-m3 score, used only when RAG_RERANK=1. CALIBRATED: the cross-encoder separates
# better than dense (balanced acc 0.897 @ 0.794 vs dense 0.76) BUT (a) still can't reject the
# hardest near-miss (death-sentences-India scored 0.987 — that's the verifier's job, not the gate's),
# and (b) over-abstains ~12% of real gold (UDHR enumerations, an FFM question at 0.013) at its
# balanced-best. So the reranker is NOT the default gate. 0.5 is a recall-preserving floor if enabled.
MIN_RERANK = 0.5
# MEASURED HARMFUL: raw BM25 magnitude rewards topical word-overlap in a large corpus, not just
# exact ids — it answered ALL 12 negatives. Override DISABLED (inf); kept parameterizable for the
# genuine exact-identifier case, but never fires with this default.
MIN_BM25 = float("inf")


def _hit_signal(h: dict):
    """Best absolute relevance signal on a hit: cross-encoder score if the reranker ran,
    else dense cosine. None when the hit carries neither (e.g. a section chunk pulled in by
    the enumerate/table dispatch rather than by search)."""
    rs = h.get("rerank_score")
    return rs if rs is not None else h.get("dense_score")


def _chunk_hit(c) -> dict:
    """Hit dict for a chunk — same shape as HybridIndex._hit (the structure-tool path)."""
    return {"text": c.text, "page": c.page_start, "printed_page": c.printed_page_start,
            "section_path": list(c.section_path), "chunk_id": c.id, "kind": c.kind,
            "doc_id": c.doc_id, "chapter": c.chapter}


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

        # Entity-filter vocabulary (Part 2): chapter names present on the chunks + province/appendix
        # aliases, so a question's "Punjab" / "KP" / "HRCP stands" can restrict retrieval to that
        # chapter. Year→doc needs no map (a doc matches a year iff the year is in its doc_id).
        from framework.ingest.shr import normalize, _PROVINCE_ALIASES
        self._chapters = {c.chapter for c in chunks if c.chapter}
        # Exclude year-bearing chapter names (the overview "… in 2024") from the FILTER vocab: they
        # are low-signal and false-match a question's own country+year phrasing ("death row in
        # Pakistan in 2024"). They remain real chapters for citations — just not filter keys.
        alias = {normalize(ch): ch for ch in self._chapters
                 if normalize(ch) and not re.search(r"\b(?:19|20)\d{2}\b", ch)}
        for a, canon in _PROVINCE_ALIASES.items():
            if canon in self._chapters:
                alias[normalize(a)] = canon
        if "HRCP stands" in self._chapters:          # natural phrasings of the appendix title
            for syn in ("hrcp stance", "hrcp position", "hrcp stand"):
                alias[syn] = "HRCP stands"
        self._alias_map = alias

    def doc_label(self, doc_id: str) -> str:
        return self._labels.get(doc_id, doc_id)

    def _allowed_ids(self, entities: dict):
        """Chunk ids the question's entities permit — docs matching a year AND chunks matching a
        chapter. None when there are no entities (no filter). The caller relaxes to unfiltered if
        the result is empty, so a mis-read entity can never cause a false 'not covered'."""
        years, chapters = entities.get("years"), entities.get("chapters")
        if not years and not chapters:
            return None
        allowed = set()
        for c in self._chunks:
            if years and not any(y in c.doc_id for y in years):
                continue
            if chapters and c.chapter not in chapters:
                continue
            allowed.add(c.id)
        return allowed

    # --- deterministic abstention gate -------------------------------------

    def confidence(self, hits: list):
        """The strongest relevance signal present in `hits` (max, not top-1 — robust to the
        enumerate/table dispatch reordering the list). None if nothing carries a signal."""
        vals = [s for h in hits if (s := _hit_signal(h)) is not None]
        return max(vals) if vals else None

    def confident(self, hits: list, *, min_dense=None, min_rerank=None, min_bm25=None) -> bool:
        """Should the served path ANSWER these hits (vs deterministically abstain)? The
        question is "does the context contain a strong match", so we judge the BEST hit.
        A strong exact lexical (BM25) match overrides a low cosine — exact ids/clauses are
        certain even when dense similarity is low. Otherwise: reranker scale if it ran, else
        dense cosine. Thresholds default to the calibrated module constants."""
        if not hits:
            return False
        mb = MIN_BM25 if min_bm25 is None else min_bm25
        if any((h.get("bm25_score") or 0.0) >= mb for h in hits):
            return True
        reranked = [h["rerank_score"] for h in hits if h.get("rerank_score") is not None]
        if reranked:
            mr = MIN_RERANK if min_rerank is None else min_rerank
            return max(reranked) >= mr
        dense = [h["dense_score"] for h in hits if h.get("dense_score") is not None]
        md = MIN_DENSE if min_dense is None else min_dense
        return bool(dense) and max(dense) >= md

    def _search(self, query: str, k: int, allowed: set | None = None) -> list:
        pool = 100 if allowed is None else 400       # widen the pool when filtering a small subset
        hits = self._index.retrieve(query, self._candidates, candidates=pool, allowed=allowed)
        if self._scorer is not None:
            from framework.rag.rerank import rerank
            return rerank(query, hits, k, scorer=self._scorer)
        return hits[:k]

    # --- structure access (Phase D dispatch targets) -----------------------

    def list_outline(self, doc_id: str) -> list:
        doc = self._docs.get(doc_id)
        return list(doc.outline) if doc else []

    def get_section(self, doc_id: str, title: str, *, budget: int = SECTION_TOKEN_BUDGET) -> list:
        """All LEAF chunks of a section (partial, case-insensitive title match) in doc
        order, accumulated up to a token budget. Leaves only (no parent/child dup).
        Returns hit dicts; empty if nothing matches."""
        from framework.context.budget import count_tokens
        from framework.rag.chunker import leaf_chunks
        t = (title or "").lower()
        if not t:
            return []
        leaves = [c for c in leaf_chunks(self._chunks)
                  if c.doc_id == doc_id and any(t in s.lower() for s in c.section_path)]
        out, used = [], 0
        for c in leaves:
            tok = count_tokens(c.text)
            if out and used + tok > budget:          # always keep at least one chunk
                break
            out.append(_chunk_hit(c))
            used += tok
        return out

    def get_table(self, doc_id: str, query: str) -> list:
        """The first atomic table chunk of `doc_id` matching `query` (keyword or section),
        verbatim. Empty if none."""
        t = (query or "").lower()
        for c in self._chunks:
            if c.kind == "table" and c.doc_id == doc_id and (
                    t in c.text.lower() or any(t in s.lower() for s in c.section_path)):
                return [_chunk_hit(c)]
        return []

    def retrieve(self, question: str, k: int) -> list:
        """Deterministic query-shape dispatch (Phase D). For an enumerate/table question,
        AUGMENT the top-k hybrid search with the whole located section/table: the search
        FIRST LOCATES the best doc+section, then get_section/get_table EXPANDS it, and the
        two are UNIONED (section-first, deduped). Union — not replace — so a misroute can
        only add noise, never drop the coverage plain search already had. Facts get top-k
        only. One deterministic retrieval, no agent loop."""
        from framework.rag.router import classify_query, extract_entities
        # Entity filter (Part 2) — HELP-ONLY. Try restricting to the doc(s)/chapter(s) the question
        # names, but keep the filtered view ONLY when it's at least as confident as the unfiltered
        # search. So a mis-read year, a province that isn't where the answer lives, or a garbage
        # chapter can never make retrieval worse (measured: a HARD filter regressed SHR 2024 r@5
        # 0.938→0.812), while a genuinely better-targeted filter (e.g. the HRCP-stands appendix) wins.
        hits = self._search(question, k)
        allowed = self._allowed_ids(extract_entities(question, self._alias_map))
        if allowed:
            fhits = self._search(question, k, allowed=allowed)
            # STRICTLY more confident, not merely equal: on a tie the filtered top-1 is the SAME
            # chunk (the filter didn't improve precision), so keeping it can only drop lower-ranked
            # relevant chunks the unfiltered search had — a pure recall loss. A distractor year
            # ("…in the previous year, 2023?" on a 2024 question) ties and is correctly left
            # unfiltered; a genuine win (the HRCP-stands appendix surfacing a higher-scoring chunk)
            # raises confidence and still takes over.
            if fhits and (self.confidence(fhits) or -1.0) > (self.confidence(hits) or -1.0):
                hits = fhits
        shape = classify_query(question)
        if shape == "fact" or not hits:
            return hits
        doc_id = hits[0]["doc_id"]
        sec = hits[0].get("section_path") or []
        title = sec[-1] if sec else ""
        extra = self.get_table(doc_id, title) if shape == "table" else self.get_section(doc_id, title)
        if not extra:
            return hits
        seen = {h["chunk_id"] for h in extra}
        return extra + [h for h in hits if h["chunk_id"] not in seen]
