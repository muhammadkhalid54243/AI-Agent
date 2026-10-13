"""Calibrate the deterministic abstention gate — the threshold is CHOSEN by measurement.

The served path abstains (NOT_COVERED, no model call) when retrieval confidence is below a
threshold. This script measures, over the full English corpus, how a candidate threshold
separates ANSWERABLE gold (should answer) from HARD NEGATIVES (should abstain), so the number
in rag/corpus.py is data-backed rather than guessed.

Retrieval only — no LLM. Each question is retrieved ONCE; thresholds are swept over the cached
confidence signals.

    uv run python examples/abstention_eval.py            # dense cosine gate (product default)
    uv run python examples/abstention_eval.py --rerank   # cross-encoder gate
"""

import glob
import os
import sys

from framework.eval.retrieval import load_gold
from framework.ingest import ingest
from framework.rag.corpus import CorpusRetriever, MIN_BM25

RAG_K = 8
ENGLISH_DIRS = ["docs/SHR", "docs/FFM", "docs/Thematic Reports"]
POS_GOLD = [
    "docs/gold/2024_SHR_English.jsonl", "docs/gold/2024_SHR_HRCP_stands.jsonl",
    "docs/gold/2023_SHR_English.jsonl", "docs/gold/FFM_police_custody.jsonl",
    "docs/gold/2025_Article9A_thematic.jsonl", "docs/gold/exhaustive_english.jsonl",
]
NEG_GOLD = ["docs/gold/unanswerable_english.jsonl"]


def _english_pdfs() -> list:
    paths = []
    for d in ENGLISH_DIRS:
        paths.extend(glob.glob(os.path.join(d, "*.pdf")))
    return sorted(dict.fromkeys(paths))


def _stats(name, vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        print(f"  {name:10} (no signal)")
        return
    mid = vals[len(vals) // 2]
    print(f"  {name:10} n={len(vals):<3} min={vals[0]:+.3f}  median={mid:+.3f}  max={vals[-1]:+.3f}")


def main() -> None:
    rerank = "--rerank" in sys.argv
    scorer = None
    if rerank:
        from framework.rag.rerank import default_scorer
        scorer = default_scorer()

    docs = [ingest(p) for p in _english_pdfs()]
    cr = CorpusRetriever(docs, scorer=scorer)
    signal = "rerank_score" if rerank else "dense cosine"
    print(f"Corpus: {len(docs)} English docs | gate signal: {signal} | BM25 override >= {MIN_BM25}\n")

    pos = [g for p in POS_GOLD for g in load_gold(p)]
    neg = [g for p in NEG_GOLD for g in load_gold(p)]
    pos_hits = {g.id: cr.retrieve(g.question, RAG_K) for g in pos}
    neg_hits = {g.id: cr.retrieve(g.question, RAG_K) for g in neg}

    print(f"Positives: {len(pos)} answerable gold   Negatives: {len(neg)} hard unanswerables\n")
    print("Confidence signal distribution (the value the threshold compares against):")
    _stats("positives", [cr.confidence(pos_hits[g.id]) for g in pos])
    _stats("negatives", [cr.confidence(neg_hits[g.id]) for g in neg])

    # Sweep the active signal. `confident` applies the SAME BM25 override the product uses, so
    # a strong exact lexical match can answer regardless of threshold — this is where topical
    # near-miss negatives (shared rare tokens) show up as false answers.
    confs = [c for g in pos for c in [cr.confidence(pos_hits[g.id])] if c is not None]
    confs += [c for g in neg for c in [cr.confidence(neg_hits[g.id])] if c is not None]
    lo, hi = (min(confs), max(confs)) if confs else (0.0, 1.0)
    grid = [round(lo + (hi - lo) * i / 24, 3) for i in range(25)]

    # MEASURED: the BM25 raw-magnitude override fires on topical word-overlap, not just exact
    # ids, so it answers every near-miss negative and defeats the gate. Disabled by default for
    # calibration; pass --bm25-override to see its (harmful) effect.
    override = "--bm25-override" in sys.argv
    def answered(hits, th):
        return cr.confident(hits, min_dense=th, min_rerank=th,
                            min_bm25=(None if override else 1e18))

    print(f"\n{'threshold':>10}  {'answered(pos)':>14}  {'abstained(neg)':>15}  {'balanced':>9}")
    print("-" * 56)
    best = None
    for th in grid:
        ans = sum(1 for g in pos if answered(pos_hits[g.id], th)) / len(pos)
        abst = sum(1 for g in neg if not answered(neg_hits[g.id], th)) / len(neg)
        bal = (ans + abst) / 2
        flag = ""
        if best is None or bal > best[1] + 1e-9:
            best, flag = (th, bal), "  <-"
        print(f"{th:>10.3f}  {ans:>14.3f}  {abst:>15.3f}  {bal:>9.3f}{flag}")

    th = best[0]
    print(f"\nBest balanced threshold: min_dense = {th}  (balanced acc {best[1]:.3f})")
    print("Negatives still ANSWERED at that threshold (gate false-positives to inspect):")
    for g in neg:
        if answered(neg_hits[g.id], th):
            h = neg_hits[g.id]
            bm = max((x.get("bm25_score") or 0.0) for x in h)
            print(f"  - {g.id:28} conf={cr.confidence(h)}  bm25_max={bm:.1f}  "
                  f"(override={'yes' if bm >= MIN_BM25 else 'no'})")
    print("Positives WRONGLY abstained at that threshold (coverage lost):")
    for g in pos:
        if not answered(pos_hits[g.id], th):
            print(f"  - {g.id:28} conf={cr.confidence(pos_hits[g.id])}")


if __name__ == "__main__":
    main()
