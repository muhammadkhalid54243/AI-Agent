"""Routing — decide whether to retrieve at all. Retrieval is LOSSY: it surfaces a
subset and throws the rest away. So if a whole document fits in the context window,
retrieving from it is strictly worse than putting it ALL in context — you'd be
discarding information you could have kept, to save tokens you don't need to save.
Prompt caching then makes the repeated full-document prefix cheap on follow-up turns.

This is why the `report` profile exists: most reports should never be chunked. A
pipeline that always retrieves is strictly worse than one that knows when not to.
"""

import re

from framework.context.budget import count_tokens

# Query SHAPE for retrieval dispatch (Phase D). Deterministic, no LLM: a "list every X"
# question needs the WHOLE section, a "table" question the whole table, a fact question
# the usual top-k. "all"/"how many" are excluded — they over-trigger on plain facts.
_ENUMERATE = re.compile(r"\b(list|every|each|enumerate|entire)\b", re.I)


def classify_query(question: str) -> str:
    """Return the query shape: 'table' | 'enumerate' | 'fact'."""
    q = question or ""
    if re.search(r"\btable\b", q, re.I):
        return "table"
    if _ENUMERATE.search(q):
        return "enumerate"
    return "fact"


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def extract_entities(question: str, alias_map: dict | None = None) -> dict:
    """Deterministically pull filterable entities from a question (no LLM):

      - `years`:    4-digit years → used to restrict retrieval to the report doc(s) for that year.
      - `chapters`: canonical chapter names whose alias appears (as a WHOLE phrase) in the question.

    `alias_map` is {normalized-alias → canonical-chapter}, built by the retriever from the corpus's
    own chapter metadata + province aliases. Matching is whole-word on the normalized question, so
    'Punjab'/'KP'/'HRCP stands' match but partial words don't. Used by CorpusRetriever to filter
    retrieval BEFORE searching; a filter only tightens (the retriever relaxes if it finds nothing).
    """
    q = question or ""
    years = set(_YEAR.findall(q))
    chapters = set()
    if alias_map:
        from framework.ingest.shr import normalize
        nq = normalize(re.sub(r"['’]s?\b", "", q))   # drop possessives: HRCP's -> HRCP
        for alias, canon in alias_map.items():
            if alias and re.search(rf"\b{re.escape(alias)}\b", nq):
                chapters.add(canon)
    return {"years": years, "chapters": chapters}


# Below this many tokens, stuff the whole document; above it, retrieve.
DEFAULT_THRESHOLD = 50_000
# Per-profile overrides — reports lean toward stuffing, big structured PDFs toward retrieval.
_PROFILE_THRESHOLD = {"report": 60_000, "structured_pdf": 50_000, "publication": 50_000}


def doc_tokens(doc) -> int:
    """Approximate token size of a ParsedDoc (its block text)."""
    return sum(count_tokens(b.text) for b in doc.blocks)


def route(doc, *, threshold: int | None = None) -> str:
    """Return "stuff" (put the whole doc in context) or "retrieve" (chunk + search)."""
    limit = threshold if threshold is not None else _PROFILE_THRESHOLD.get(doc.profile, DEFAULT_THRESHOLD)
    return "stuff" if doc_tokens(doc) <= limit else "retrieve"
