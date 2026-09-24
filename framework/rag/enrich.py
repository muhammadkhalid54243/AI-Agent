"""Enrichment — populate `Chunk.embed_text` (what is EMBEDDED); never touch
`Chunk.text` (what is RETURNED to the model).

Keeping the two separate is the load-bearing rule: what makes a chunk *findable*
(a breadcrumb, expanded acronyms, a situating sentence) is not what should be *shown*
to the model. If you embed AND return that scaffolding, the model starts citing your
machine-written metadata as though it were the document. So enrichment only rewrites
`embed_text`; the author's words in `text` stay untouched.

Two deterministic, free enrichers run by default:
  - breadcrumb: prepend the section path, so a chunk carries where it lives.
  - glossary:   for each abbreviation the chunk mentions, append `ACR (Expansion)`,
                so a query using the spelled-out term matches the chunk that only
                used the acronym (the Phase-0 PECA vocabulary miss).

The contextual prefix (one cheap-LLM call per chunk) is opt-in and cost-controlled;
see `contextual_prefixes`.
"""

import re


def _mentions(text: str, acronym: str) -> bool:
    return re.search(rf"\b{re.escape(acronym)}\b", text) is not None


def enrich_chunks(chunks: list, *, glossary: dict | None = None, breadcrumb: bool = True,
                  use_glossary: bool = True, prefixes: dict | None = None,
                  max_glossary_freq: int = 8) -> list:
    """Rewrite each chunk's `embed_text` in place from its `text` + metadata.

    glossary: {acronym: GlossaryEntry} (from `ParsedDoc.glossary`).
    prefixes: optional {chunk_id: contextual sentence} to prepend (see contextual_prefixes).
    max_glossary_freq: only expand acronyms that appear in <= this many chunks. A
      UBIQUITOUS acronym (e.g. the publisher's, in hundreds of chunks) appended
      everywhere becomes non-discriminative and HURTS retrieval of its own definition
      (measured: HRCP fell out of top-20) — so frequency-gate it out. Rare acronyms
      (PECA, IDP) still get expanded, where the expansion actually helps.
    Idempotent: always rebuilt from `text`, so re-running never double-appends.
    """
    glossary = glossary or {}
    prefixes = prefixes or {}

    # Count how many chunks mention each acronym; only expand the rare ones.
    rare = set()
    if use_glossary and glossary:
        freq = {a: 0 for a in glossary}
        for c in chunks:
            if c.kind == "table":
                continue
            for a in glossary:
                if _mentions(c.text, a):
                    freq[a] += 1
        rare = {a for a, n in freq.items() if 0 < n <= max_glossary_freq}

    for c in chunks:
        parts = []
        if c.id in prefixes and prefixes[c.id]:
            parts.append(prefixes[c.id].strip())
        if breadcrumb and c.section_path:
            parts.append(" > ".join(c.section_path))
        parts.append(c.text)
        if rare and c.kind != "table":
            hits = [f"{a} ({glossary[a].expansion})" for a in rare
                    if glossary[a].expansion and _mentions(c.text, a)]
            if hits:
                parts.append("Abbreviations: " + "; ".join(hits))
        c.embed_text = "\n\n".join(parts)
    return chunks


CONTEXT_PROMPT = (
    "Here is the whole document for context, then one chunk from it. In 1-2 sentences, "
    "state what this chunk is about and where it sits in the document. Output ONLY those "
    "sentences.\n\n<document>\n{doc}\n</document>\n\n<chunk>\n{chunk}\n</chunk>"
)


def contextual_prefixes(chunks: list, doc_text: str, *, summarizer, max_calls: int = 50,
                        cache: dict | None = None) -> dict:
    """Opt-in: one cheap-LLM situating sentence per chunk (Anthropic 'contextual
    retrieval'). Returns {chunk_id: sentence}. Hard-capped and cache-backed so a
    re-run is free; `summarizer` is injectable (a fake in tests, a real model live).
    Errors on a chunk degrade to no prefix, never abort."""
    cache = cache if cache is not None else {}
    out = {}
    calls = 0
    for c in chunks:
        if c.id in cache:
            out[c.id] = cache[c.id]
            continue
        if calls >= max_calls:
            break
        try:
            sentence = summarizer(CONTEXT_PROMPT.format(doc=doc_text, chunk=c.text)).strip()
        except Exception:
            sentence = ""
        calls += 1
        cache[c.id] = sentence
        if sentence:
            out[c.id] = sentence
    return out
