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


def test_completeness_fraction_of_context_and_excluded_from_recall():
    gold = [GoldItem("ex", "list all", kind="exhaustive",
                     expected_items=["alpha", "bravo", "charlie", "delta"])]
    # assembled context contains alpha + charlie only -> 2/4 = 0.5
    ranked = {"list all": [{"text": "ALPHA and CHARLIE appear"}, {"text": "noise"}]}
    r = score_retrieval(_fake_retrieve(ranked), gold)
    assert r["completeness"] == 0.5          # fraction of items in the assembled context
    assert r["completeness_scored"] == 1
    assert r["n"] == 0                        # exhaustive excluded from single-fact metrics
    assert r["recall"]["@5"] == 0.0          # recall@k is blind to the enumeration


def test_completeness_is_doc_gated():
    gold = [GoldItem("ex", "list", kind="exhaustive", gold_doc_id="D1",
                     expected_items=["alpha", "bravo"])]
    ranked = {"list": [
        {"text": "ALPHA", "doc_id": "D1"},   # counts
        {"text": "BRAVO", "doc_id": "D2"},   # identical-looking WRONG doc -> ignored
    ]}
    r = score_retrieval(_fake_retrieve(ranked), gold)
    assert r["completeness"] == 0.5          # only alpha (D1) counts


def test_completeness_respects_context_k_depth():
    # The blind spot in one test: at a shallow context depth the enumeration is incomplete;
    # a dispatch that returns the whole section (more chunks than context_k) escapes it.
    gold = [GoldItem("ex", "list", kind="exhaustive", expected_items=["a", "b", "c"])]
    ranked = {"list": [{"text": "A"}, {"text": "B"}, {"text": "C"}]}
    assert score_retrieval(_fake_retrieve(ranked), gold, context_k=2)["completeness"] == round(2 / 3, 3)
    assert score_retrieval(_fake_retrieve(ranked), gold, context_k=8)["completeness"] == 1.0


def test_abstention_metrics_measure_the_gate():
    # A perfect gate: confident on the answerable question, not on the unanswerable one.
    gold = [GoldItem("a", "real question", answer_phrases=["x"]),
            GoldItem("n", "nonsense question", kind="unanswerable")]
    ranked = {"real question": [{"text": "x marks it"}], "nonsense question": []}
    is_conf = lambda q: q == "real question"
    r = score_retrieval(_fake_retrieve(ranked), gold, is_confident=is_conf)
    assert r["abstain_rate"] == 1.0          # abstained on the one unanswerable
    assert r["false_abstain_rate"] == 0.0    # did not abstain on the answerable
    assert r["abstain_scored"] == 1


def test_abstention_metrics_none_without_gate():
    gold = [GoldItem("n", "q", kind="unanswerable")]
    r = score_retrieval(_fake_retrieve({"q": []}), gold)
    assert r["abstain_rate"] is None and r["false_abstain_rate"] is None


def test_false_abstain_counts_a_wrongly_gated_answerable():
    gold = [GoldItem("a", "qa", answer_phrases=["x"]), GoldItem("b", "qb", answer_phrases=["y"])]
    ranked = {"qa": [{"text": "x"}], "qb": [{"text": "y"}]}
    is_conf = lambda q: q == "qa"            # wrongly unsure about qb
    r = score_retrieval(_fake_retrieve(ranked), gold, is_confident=is_conf)
    assert r["false_abstain_rate"] == 0.5    # 1 of 2 answerables wrongly abstained
