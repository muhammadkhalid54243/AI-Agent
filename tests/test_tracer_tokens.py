from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from framework.observability import TrajectoryTracer


def test_tracer_records_context_token_breakdown():
    t = TrajectoryTracer()
    msgs = [[SystemMessage(content="You are helpful."),
             HumanMessage(content="What is the weather in Lahore?")]]
    t.on_chat_model_start({}, msgs, run_id="a")
    t.on_llm_end(_fake_llm_result(input_tokens=25, output_tokens=10), run_id="a")

    rep = t.token_report()
    assert rep["peak_context_tokens"] > 0
    assert rep["input_tokens"] == 25
    assert rep["output_tokens"] == 10
    # the recorded model-call event carries a per-section breakdown
    ev = next(e for e in t.events if e["kind"] == "model_call")
    assert ev["context"]["system"] > 0
    assert ev["context"]["history"] > 0


def test_tracer_reports_cache_tokens():
    t = TrajectoryTracer()
    msgs = [[SystemMessage(content="big stable prefix")]]
    t.on_chat_model_start({}, msgs, run_id="c")
    usage = {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
             "input_token_details": {"cache_read": 900, "cache_creation": 0}}
    t.on_llm_end(_Result(_Msg(usage)), run_id="c")
    rep = t.token_report()
    assert rep["cache_read_tokens"] == 900
    assert rep["cache_write_tokens"] == 0


class _Msg:
    def __init__(self, usage):
        self.usage_metadata = usage


class _Gen:
    def __init__(self, msg):
        self.message = msg


class _Result:
    def __init__(self, msg):
        self.generations = [[_Gen(msg)]]
        self.llm_output = None


def _fake_llm_result(input_tokens, output_tokens):
    return _Result(_Msg({"input_tokens": input_tokens, "output_tokens": output_tokens,
                         "total_tokens": input_tokens + output_tokens}))
