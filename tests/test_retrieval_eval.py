"""Phase 0 scorer — deterministic, no API key, no PDF."""

from framework.eval.retrieval import GoldItem, is_relevant, score_retrieval


def _fake_retrieve(ranked_by_q):
    """ranked_by_q: {question: [hit, ...]} -> a retrieve(query, k) callable."""
    def retrieve(query, k):
        return ranked_by_q.get(query, [])[:k]
    return retrieve


def test_is_relevant_matches_any_signal():
    g = GoldItem("q1", "?", gold_pages={7}, gold_sections={"4.2 Losses"},
                 answer_phrases=["power purchase"])
    assert is_relevant({"page": 7}, g)                              # page
    assert is_relevant({"section_path": ("4. T", "4.2 Losses")}, g)  # section
    assert is_relevant({"text": "the Power Purchase Agreement"}, g)  # phrase (ci)
    assert not is_relevant({"page": 8, "text": "unrelated"}, g)


def test_recall_mrr_and_answer_hit():
    gold = [
        GoldItem("a", "qa", answer_phrases=["alpha"]),
        GoldItem("b", "qb", answer_phrases=["bravo"]),
    ]
    ranked = {
        # first relevant at rank 2  -> in @5 and @20, RR=0.5
        "qa": [{"text": "noise"}, {"text": "ALPHA here"}, {"text": "x"}],
        # relevant only at rank 9   -> in @20 not @5, RR=1/9
        "qb": [{"text": "n"}] * 8 + [{"text": "bravo!"}],
    }
    r = score_retrieval(_fake_retrieve(ranked), gold, ks=(5, 20))
    assert r["recall"]["@5"] == 0.5      # only qa within top-5
    assert r["recall"]["@20"] == 1.0     # both within top-20
    assert r["mrr"] == round((0.5 + 1 / 9) / 2, 3)
    assert r["answer_hit_rate"] == 1.0   # both phrases appear somewhere in top-k
    assert r["citation_accuracy"] is None  # no pages in these hits


def test_citation_accuracy_scored_only_with_pages():
    gold = [GoldItem("a", "qa", gold_pages={3}), GoldItem("b", "qb", gold_pages={9})]
    ranked = {
        "qa": [{"page": 3, "text": "right page"}],   # top-1 correct
        "qb": [{"page": 5, "text": "wrong page"}],   # top-1 wrong
    }
    r = score_retrieval(_fake_retrieve(ranked), gold)
    assert r["citation_scored"] == 2
    assert r["citation_accuracy"] == 0.5


def test_total_miss_scores_zero():
    gold = [GoldItem("a", "qa", answer_phrases=["never"])]
    r = score_retrieval(_fake_retrieve({"qa": [{"text": "nope"}]}), gold)
    assert r["recall"]["@5"] == 0.0 and r["mrr"] == 0.0
    assert r["rows"][0]["first_relevant_rank"] is None
