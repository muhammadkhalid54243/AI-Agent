"""Cross-document-type generalization eval.

Build ONE corpus spanning three document shapes — annual SHR (2023 + 2024), an FFM
fact-finding report, and a thematic policy paper — and score each gold set against the
WHOLE corpus. What to read:
  - recall@5 per set: is each document type retrievable inside a mixed corpus?
  - doc_attribution: cross-TYPE should be high (topics differ, little confusion);
    same-series SHR year-attribution stays mixed BY DESIGN — the cited-answer format
    (each fact on its own line, labelled with its year) handles that, not a routing gate.

    uv run python examples/corpus_eval.py            # hybrid only
    uv run python examples/corpus_eval.py --rerank   # + cross-encoder
"""

import sys

from framework.eval.retrieval import load_gold, score_retrieval
from framework.ingest import ingest
from framework.rag.corpus import CorpusRetriever

DOCS = [
    "docs/SHR/2023_SHR_English.pdf",
    "docs/SHR/2024_SHR_English.pdf",
    "docs/FFM/2026-Deaths-in-police-custody-in-Lahore.pdf",
    "docs/Thematic Reports/2025-The-promise-of-Article-9A.pdf",
]
GOLD = {
    "SHR 2024": ["docs/gold/2024_SHR_English.jsonl", "docs/gold/2024_SHR_HRCP_stands.jsonl"],
    "SHR 2023": ["docs/gold/2023_SHR_English.jsonl"],
    "FFM police custody": ["docs/gold/FFM_police_custody.jsonl"],
    "Thematic Article 9A": ["docs/gold/2025_Article9A_thematic.jsonl"],
}


def main() -> None:
    rerank = "--rerank" in sys.argv
    scorer = None
    if rerank:
        from framework.rag.rerank import default_scorer
        scorer = default_scorer()
    docs = [ingest(p) for p in DOCS]
    cr = CorpusRetriever(docs, scorer=scorer)
    print(f"Corpus: {', '.join(cr.doc_label(d.doc_id) for d in docs)}"
          f"{' [rerank]' if rerank else ''}\n")
    hdr = f"{'gold set':22} {'n':>3}  r@5    r@20   MRR    ans    doc_attr"
    print(hdr)
    print("-" * len(hdr))
    for label, paths in GOLD.items():
        gold = []
        for p in paths:
            gold += load_gold(p)
        res = score_retrieval(lambda q, k: cr._search(q, k), gold)
        print(f"{label:22} {res['n']:>3}  {str(res['recall']['@5']):<6} "
              f"{str(res['recall']['@20']):<6} {str(res['mrr']):<6} "
              f"{str(res['answer_hit_rate']):<6} {res['doc_attribution']}")


if __name__ == "__main__":
    main()
