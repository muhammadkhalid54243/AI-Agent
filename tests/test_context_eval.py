from framework.eval.context import score_context


class FakeAgent:
    """Returns canned run_traced results so the aggregation is tested without a network."""
    def __init__(self, results):
        self._results = list(results)

    def run_traced(self, message, thread_id=None):
        return self._results.pop(0)


def _result(inp, out, peak, calls, tools):
    return {
        "answer": "ok",
        "summary": {"model_calls": calls, "tool_calls": tools, "total_ms": 100},
        "tokens": {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                   "peak_context_tokens": peak,
                   "context_by_section": {"system": 50, "history": 100, "tool_results": peak - 150}},
        "events": [],
    }


def test_score_context_aggregates_and_costs():
    agent = FakeAgent([_result(1000, 200, 1200, 3, 2), _result(500, 100, 700, 2, 1)])
    report = score_context(agent, ["task one", "task two"],
                           price_per_mtok={"input": 1.0, "output": 2.0})
    assert len(report["rows"]) == 2
    assert report["totals"]["total_tokens"] == 1800
    assert report["totals"]["peak_context"] == 1200
    # cost: input (1500/1e6 * 1.0) + output (300/1e6 * 2.0)
    assert report["totals"]["cost_usd"] == round(1500 / 1e6 * 1.0 + 300 / 1e6 * 2.0, 6)
