"""Multi-turn conversation history — offline, no model."""

from framework.serve.history import History


def test_get_is_empty_for_unknown_thread():
    assert History().get("nope") == []


def test_add_and_render_round_trips():
    h = History()
    h.add("t1", "How many in 2024?", "6,161 [SHR 2024]")
    h.add("t1", "and 2023?", "3,800 [SHR 2023]")
    rendered = h.render("t1")
    assert "User: How many in 2024?" in rendered
    assert "Assistant: 6,161 [SHR 2024]" in rendered
    assert "and 2023?" in rendered


def test_threads_are_isolated():
    h = History()
    h.add("a", "q-a", "ans-a")
    h.add("b", "q-b", "ans-b")
    assert "q-a" in h.render("a") and "q-a" not in h.render("b")


def test_caps_to_max_turns():
    h = History(max_turns=2, max_tokens=10_000)
    for i in range(5):
        h.add("t", f"q{i}", f"a{i}")
    assert len(h.get("t")) == 2
    assert h.get("t")[0][0] == "q3" and h.get("t")[1][0] == "q4"   # newest kept


def test_caps_to_token_budget():
    h = History(max_turns=100, max_tokens=20)       # tiny budget forces eviction
    for i in range(10):
        h.add("t", f"question number {i}", f"a long-ish answer number {i}")
    # Oldest turns evicted until under budget; at least the most recent survives.
    turns = h.get("t")
    assert 1 <= len(turns) < 10
    assert turns[-1][0] == "question number 9"
