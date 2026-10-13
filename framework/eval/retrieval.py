"""Retrieval scoring — measure the INDEX, separately from answer quality.

Phase 0 of the ingestion track. A bad answer is either a RETRIEVAL miss (the right
chunk never surfaced) or a GENERATION miss (it surfaced and the model ignored it).
Fusing those two signals makes the system permanently undebuggable, so this scores
retrieval on its own — no LLM in the loop.

A GoldItem declares what SHOULD surface for a question, by any of three signals:
page, section, or an answer phrase that must appear in a retrieved chunk. A hit is
"relevant" if it matches any declared signal — so the SAME gold set works before
metadata exists (text-only baseline, phrase match) and after Phase 1-2 add page /
section metadata (page/section match). Metrics:

  - recall@k          did ≥1 relevant chunk make the top k? (k=5 ranking, k=20 index)
  - MRR               1/rank of the first relevant chunk, averaged
  - answer_hit_rate   does the answer text appear ANYWHERE in the top-k? (a
                      generation-independent "could it even answer" probe)
  - citation_accuracy top-1 page ∈ gold_pages — only scored once chunks carry pages;
                      None at baseline (bare chunks have no page), by design.
  - completeness      EXHAUSTIVE items only: fraction of expected_items present in the
                      CONTEXT the product assembles (doc-gated retrieve(q, context_k), the
                      WHOLE returned context — not top-k-sliced, since a whole-section
                      dispatch isn't a ranked list). recall@k fires on ONE relevant chunk,
                      so it is blind to enumeration; completeness is not.
"""

from dataclasses import dataclass, field


@dataclass
class GoldItem:
    id: str
    question: str
    answer: str = ""
    gold_pages: set = field(default_factory=set)
    gold_sections: set = field(default_factory=set)
    answer_phrases: list = field(default_factory=list)  # any-of; falls back to `answer`
    kind: str = ""
    gold_doc_id: str = ""            # which document the answer must come from (multi-doc)
    expected_items: list = field(default_factory=list)  # exhaustive: ALL must appear (completeness)

    def phrases(self) -> list:
        return self.answer_phrases or ([self.answer] if self.answer else [])


def _section_str(sec) -> str:
    if isinstance(sec, (list, tuple)):
        return " > ".join(str(s) for s in sec)
    return str(sec or "")


def is_relevant(hit: dict, gold: GoldItem) -> bool:
    """A hit is relevant if it matches ANY declared gold signal (page/section/phrase).

    Multi-doc gate: if the gold item names a `gold_doc_id` and the hit carries a
    `doc_id`, they must match first — otherwise a same-named section from the WRONG
    report (every SHR year has "Key issues") would count as a hit and hide cross-year
    confusion. Hits without a doc_id (single-doc baseline) skip the gate.
    """
    if gold.gold_doc_id:
        hd = hit.get("doc_id")
        if hd is not None and hd != gold.gold_doc_id:
            return False
    page = hit.get("page")
    if gold.gold_pages and page is not None and page in gold.gold_pages:
        return True
    if gold.gold_sections:
        sec = _section_str(hit.get("section_path"))
        if sec and any(g.lower() in sec.lower() for g in gold.gold_sections):
            return True
    text = (hit.get("text") or "").lower()
    if text and any(p.lower() in text for p in gold.phrases() if p):
        return True
    return False


def score_retrieval(retrieve, gold, *, ks=(5, 20), context_k=8, is_confident=None) -> dict:
    """Score a retriever against a gold set.

    retrieve: callable(query: str, k: int) -> ranked list[hit dict]; a hit exposes
              any of {"text", "page", "section_path"}.
    gold:     list[GoldItem].
    is_confident: optional callable(question) -> bool — the deterministic abstention gate.
              When given, two abstention metrics are added: `abstain_rate` (fraction of
              `unanswerable` items the gate correctly abstains on — want HIGH) and
              `false_abstain_rate` (fraction of ANSWERABLE items it wrongly abstains on —
              want ~0). None when not supplied, so existing callers are unaffected.
    """
    gold = list(gold)
    ks = tuple(sorted(ks))
    maxk = max(ks) if ks else 20
    recall = {k: 0 for k in ks}
    rr_sum = 0.0
    answer_hits = 0
    cite_ok = cite_total = 0
    attr_ok = attr_total = 0        # doc-attribution: is top-1 from the RIGHT document?
    comp_sum = 0.0                   # completeness accumulator (exhaustive items only)
    rows = []

    # Unanswerable questions test ABSTENTION (an answer-eval concern) and exhaustive
    # questions test ENUMERATION (measured by completeness below). Both are excluded from
    # the single-fact metrics: recall@k fires on ONE relevant chunk, so it neither has a
    # target for unanswerable nor can see whether the UNION of many chunks was retrieved.
    scored = [g for g in gold if g.kind not in ("unanswerable", "exhaustive")]
    exhaustive = [g for g in gold if g.kind == "exhaustive"]
    n_unanswerable = sum(1 for g in gold if g.kind == "unanswerable")

    for g in scored:
        hits = list(retrieve(g.question, maxk) or [])
        first = next((i + 1 for i, h in enumerate(hits) if is_relevant(h, g)), None)
        for k in ks:
            if first is not None and first <= k:
                recall[k] += 1
        rr_sum += (1.0 / first) if first else 0.0

        blob = " ".join((h.get("text") or "") for h in hits[:maxk]).lower()
        phrases = g.phrases()
        if phrases and any(p.lower() in blob for p in phrases if p):
            answer_hits += 1

        # Citation accuracy is only meaningful once chunks carry a page. At baseline
        # (bare chunks) top-1 has no page → not scored, and reported as None.
        if hits and g.gold_pages and hits[0].get("page") is not None:
            cite_total += 1
            if hits[0]["page"] in g.gold_pages:
                cite_ok += 1

        # Doc-attribution: only meaningful when gold names a doc and hits carry doc_id.
        if g.gold_doc_id and hits and hits[0].get("doc_id") is not None:
            attr_total += 1
            if hits[0]["doc_id"] == g.gold_doc_id:
                attr_ok += 1

        rows.append({"id": g.id, "kind": g.kind, "first_relevant_rank": first})

    # Completeness (exhaustive items): fraction of expected_items present in the CONTEXT the
    # product actually assembles — the doc-gated FULL output of retrieve(q, context_k). Not
    # top-k-sliced: a dispatch that returns a whole section (Phase D) is not a ranked list, so
    # slicing would misread it. Doc-gating stops identical cross-year text (the UDHR appendix,
    # verbatim across SHR years) from satisfying a question about one specific year.
    for g in exhaustive:
        ctx = list(retrieve(g.question, context_k) or [])
        if g.gold_doc_id:
            ctx = [h for h in ctx if h.get("doc_id") in (None, g.gold_doc_id)]
        items = [it.lower() for it in g.expected_items if it]
        blob = " ".join((h.get("text") or "") for h in ctx).lower()
        frac = (sum(1 for it in items if it in blob) / len(items)) if items else 0.0
        comp_sum += frac
        rows.append({"id": g.id, "kind": g.kind, "completeness": round(frac, 3),
                     "context_chunks": len(ctx)})

    # Abstention (needs the gate): abstain correctly on unanswerable, answer on the rest.
    abstain_unans = abstain_ans = n_ans_for_abstain = 0
    if is_confident is not None:
        for g in gold:
            answerable_here = g.kind != "unanswerable"
            confident = bool(is_confident(g.question))
            if answerable_here:
                n_ans_for_abstain += 1
                if not confident:
                    abstain_ans += 1
            elif not confident:
                abstain_unans += 1

    for g in gold:
        if g.kind == "unanswerable":
            abstained = (not bool(is_confident(g.question))) if is_confident is not None else None
            rows.append({"id": g.id, "kind": g.kind, "abstained": abstained})

    n = len(scored) or 1
    return {
        "n": len(scored),
        "n_unanswerable": n_unanswerable,
        "recall": {f"@{k}": round(recall[k] / n, 3) for k in ks},
        "mrr": round(rr_sum / n, 3),
        "answer_hit_rate": round(answer_hits / n, 3),
        "citation_accuracy": (round(cite_ok / cite_total, 3) if cite_total else None),
        "citation_scored": cite_total,
        "doc_attribution": (round(attr_ok / attr_total, 3) if attr_total else None),
        "doc_attribution_scored": attr_total,
        "completeness": (round(comp_sum / len(exhaustive), 3) if exhaustive else None),
        "completeness_scored": len(exhaustive),
        "completeness_context_k": context_k,
        "abstain_rate": (round(abstain_unans / n_unanswerable, 3)
                         if (is_confident is not None and n_unanswerable) else None),
        "false_abstain_rate": (round(abstain_ans / n_ans_for_abstain, 3)
                               if (is_confident is not None and n_ans_for_abstain) else None),
        "abstain_scored": (n_unanswerable if is_confident is not None else 0),
        "rows": rows,
    }


def retriever_fn(retriever):
    """Adapt a framework `Retriever` into a retrieve(query, k) callable for scoring.

    Surfaces whatever metadata the store carries — page/section appear once Phase 1-2
    populate them; today only `idx` and text exist, which is the honest baseline.
    """
    def retrieve(query: str, k: int) -> list:
        docs = retriever._store.similarity_search(query, k=k)
        return [{"text": d.page_content,
                 "page": d.metadata.get("page"),
                 "section_path": d.metadata.get("section_path"),
                 "idx": d.metadata.get("idx")} for d in docs]
    return retrieve


def load_gold(path: str) -> list:
    """Load a JSONL gold file into GoldItems. Sets deserialize from lists."""
    import json
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(GoldItem(
                id=d["id"], question=d["question"], answer=d.get("answer", ""),
                gold_pages=set(d.get("gold_pages", [])),
                gold_sections=set(d.get("gold_sections", [])),
                answer_phrases=d.get("answer_phrases", []), kind=d.get("kind", ""),
                gold_doc_id=d.get("doc_id", ""),
                expected_items=d.get("expected_items", []),
            ))
    return items
