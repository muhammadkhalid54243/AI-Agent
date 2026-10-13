"""Reranking — retrieve wide, rank narrow. A cross-encoder reads the query and a
candidate TOGETHER (not two pre-computed vectors compared after the fact), so it
judges relevance far more accurately — but it is far too slow to run corpus-wide.
So the pipeline is: hybrid-retrieve ~50-100 candidates, then rerank them to the
final 5-10. This is the direct lever for the recall@5 / MRR headroom Phase 3 left
(including the ubiquitous-term HRCP case).

The scorer is injectable: a fake in tests (no model), the local BGE cross-encoder
`BAAI/bge-reranker-v2-m3` live. A scoring failure degrades to the input order.
"""


def default_scorer():
    """Lazy local cross-encoder scorer: (query, text) pairs -> relevance scores."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return lambda pairs: list(model.predict(pairs))


def rerank(query: str, hits: list, k: int, *, scorer=None) -> list:
    """Reorder `hits` by joint query-candidate relevance; return the top `k`."""
    if not hits:
        return hits
    scorer = scorer or default_scorer()
    try:
        scores = scorer([(query, h["text"]) for h in hits])
    except Exception:
        return hits[:k]                       # degrade to input order, never abort
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:k]:
        h = dict(hits[i])
        h["rerank_score"] = float(scores[i])  # cross-encoder relevance — the gate prefers this
        out.append(h)
    return out
