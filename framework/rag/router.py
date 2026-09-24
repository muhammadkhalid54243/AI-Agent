"""Routing — decide whether to retrieve at all. Retrieval is LOSSY: it surfaces a
subset and throws the rest away. So if a whole document fits in the context window,
retrieving from it is strictly worse than putting it ALL in context — you'd be
discarding information you could have kept, to save tokens you don't need to save.
Prompt caching then makes the repeated full-document prefix cheap on follow-up turns.

This is why the `report` profile exists: most reports should never be chunked. A
pipeline that always retrieves is strictly worse than one that knows when not to.
"""

from framework.context.budget import count_tokens

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
