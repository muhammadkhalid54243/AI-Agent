"""Multi-doc retrieval measurement — does adding documents dilute retrieval?

The honest before/after that gates the multi-doc design. Score the SAME 2024 SHR gold
sets against a growing corpus:

  (1) baseline:    [2024 SHR]              — single doc, the Phase-6 numbers
  (2) +distractor: [2023 SHR, 2024 SHR]    — one near-duplicate year competing
  (3) full:        [2019..2025 SHR]        — all annual years (with --all)

Two numbers decide it:
  - recall@5        did the right passage still make the top 5 with distractors present?
  - doc_attribution was the top-1 hit from the RIGHT year, or did a near-duplicate steal
                    it? (only meaningful because gold items now carry a doc_id)

If recall@5 holds and doc_attribution stays near 1.0 → ship the merged index as-is. If a
near-duplicate year steals hits → add the year/metadata prepend into embed_text, then a
light pre-retrieval query parse. Uses the real bge-m3 model, so run it deliberately.

    uv run python examples/multidoc_eval.py            # baseline vs +2023
    uv run python examples/multidoc_eval.py --all      # also all SHR years
    uv run python examples/multidoc_eval.py --rerank   # with the cross-encoder
"""

import sys

from framework.eval.retrieval import load_gold, score_retrieval
from framework.ingest import ingest
from framework.rag.corpus import CorpusRetriever

SHR = "docs/SHR/{}_SHR_English.pdf"
GOLD = ["docs/gold/2024_SHR_English.jsonl", "docs/gold/2024_SHR_HRCP_stands.jsonl"]


def _gold() -> list:
    items = []
    for path in GOLD:
        items += load_gold(path)
    return items


def _score(label: str, years: list, gold: list, *, rerank: bool) -> dict:
    docs = [ingest(SHR.format(y)) for y in years]
    scorer = None
    if rerank:
        from framework.rag.rerank import default_scorer
        scorer = default_scorer()
    cr = CorpusRetriever(docs, scorer=scorer)
    res = score_retrieval(lambda q, k: cr._search(q, k), gold)
    labels = ", ".join(cr.doc_label(d.doc_id) for d in docs)
    print(f"\n== {label}  [{labels}] ==")
    print(f"   recall@5={res['recall']['@5']}  recall@20={res['recall']['@20']}  "
          f"MRR={res['mrr']}  answer_hit={res['answer_hit_rate']}  "
          f"doc_attribution={res['doc_attribution']}  (n={res['n']})")
    return res


def main() -> None:
    rerank = "--rerank" in sys.argv
    gold = _gold()
    print(f"Scoring {len(gold)} gold questions (2024 SHR) against a growing corpus"
          f"{' [rerank on]' if rerank else ''}...")
    _score("baseline: 2024 only", ["2024"], gold, rerank=rerank)
    _score("+distractor: 2023 + 2024", ["2023", "2024"], gold, rerank=rerank)
    if "--all" in sys.argv:
        _score("full: all SHR years",
               ["2019", "2020", "2021", "2022", "2023", "2024", "2025"], gold, rerank=rerank)


if __name__ == "__main__":
    main()
