from framework.context.budget import count_tokens, measure_messages, TokenBudget


def test_count_tokens_nonzero_for_text_and_zero_for_empty():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_measure_messages_attributes_by_section():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Lahore?"},
        {"role": "assistant", "content": "Let me check."},
        {"role": "tool", "content": '{"city":"Lahore","temp_c":42,"condition":"sunny"}'},
    ]
    m = measure_messages(messages)
    assert m["system"] > 0
    assert m["history"] > 0
    assert m["tool_results"] > 0
    assert m["total"] == m["system"] + m["history"] + m["tool_results"]


def test_budget_records_turns_and_reports_peak():
    b = TokenBudget()
    b.record([{"role": "system", "content": "short"}])
    b.record([{"role": "system", "content": "short"},
              {"role": "tool", "content": "x" * 400}])
    assert len(b.turns) == 2
    assert b.peak_total() == b.turns[1]["total"]
    assert b.peak_total() > b.turns[0]["total"]
