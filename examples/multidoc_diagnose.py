"""Diagnose multi-doc misattribution: for each 2024 gold question, is the top-1 hit from
the right year, and does the QUESTION itself name a year? Separates a query-ambiguity
problem (no year stated -> retrieval can't know) from a true retrieval-signal problem."""

import re

from framework.eval.retrieval import load_gold
from framework.ingest import ingest
from framework.rag.corpus import CorpusRetriever

GOLD = ["docs/gold/2024_SHR_English.jsonl", "docs/gold/2024_SHR_HRCP_stands.jsonl"]

docs = [ingest("docs/SHR/2023_SHR_English.pdf"), ingest("docs/SHR/2024_SHR_English.pdf")]
cr = CorpusRetriever(docs)

gold = []
for p in GOLD:
    gold += load_gold(p)

miss = miss_year_in_q = 0
for g in gold:
    if g.kind == "unanswerable":
        continue
    hits = cr._search(g.question, 5)
    top = hits[0]["doc_id"] if hits else "NONE"
    ok = top == g.gold_doc_id
    year_in_q = bool(re.search(r"\b20\d{2}\b", g.question))
    if not ok:
        miss += 1
        miss_year_in_q += year_in_q
    flag = "OK  " if ok else "MISS"
    print(f"{flag} top={top:20} yearInQ={int(year_in_q)} {g.kind:14} {g.id}: {g.question[:66]}")

print(f"\n{miss} misattributed; of those, {miss_year_in_q} DID name a year in the question "
      f"(a real retrieval miss), {miss - miss_year_in_q} did NOT (query is year-ambiguous).")
