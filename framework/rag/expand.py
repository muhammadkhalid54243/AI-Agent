"""Child -> parent expansion. We MATCH on small children (precise embeddings) but
ANSWER from big parents (whole-section context) — the small-to-big payoff. After
retrieval + rerank, swap each surviving child for its parent section and deduplicate,
so several children of the same section collapse to one parent handed to the model.
"""


def expand_to_parents(hits: list, chunks: list) -> list:
    """Replace each hit's child chunk with its parent (if any) and dedupe by target id.
    A hit whose chunk has no parent (a lone leaf or a parent already) passes through."""
    by_id = {c.id: c for c in chunks}
    out, seen = [], set()
    for h in hits:
        child = by_id.get(h["chunk_id"])
        target = by_id.get(child.parent_id) if (child and child.parent_id) else child
        tid = target.id if target else h["chunk_id"]
        if tid in seen:
            continue
        seen.add(tid)
        if target is None:
            out.append(h)
        else:
            out.append({"text": target.text, "page": target.page_start,
                        "section_path": list(target.section_path), "chunk_id": target.id,
                        "kind": target.kind})
    return out
