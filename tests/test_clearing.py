from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from framework.context.clearing import clear_tool_results
from framework.context.budget import measure_messages


def _convo(payload=400):
    big = "x" * payload
    return [
        SystemMessage(content="sys"),
        HumanMessage(content="do stuff"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Lahore"}, "id": "1"}]),
        ToolMessage(content=big, tool_call_id="1", name="get_weather"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "London"}, "id": "2"}]),
        ToolMessage(content=big, tool_call_id="2", name="get_weather"),
        AIMessage(content="", tool_calls=[{"name": "delete_record", "args": {"record_id": "9"}, "id": "3"}]),
        ToolMessage(content=big, tool_call_id="3", name="delete_record"),
        AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "4"}]),
        ToolMessage(content=big, tool_call_id="4", name="get_weather"),
    ]


def _by_id(msgs, tcid):
    return next(m for m in msgs if getattr(m, "tool_call_id", None) == tcid)


def test_under_threshold_returns_unchanged():
    msgs = _convo(5)
    assert clear_tool_results(msgs, threshold_tokens=100000) is msgs


def test_clears_old_keeps_recent_and_nonclearable_and_tooluse():
    msgs = _convo(400)
    before = measure_messages(msgs)["total"]
    out = clear_tool_results(msgs, threshold_tokens=50, keep_recent=1,
                             non_clearable={"delete_record"})
    after = measure_messages(out)["total"]

    assert after < before                                   # context shrank
    assert _by_id(out, "1").content.startswith("[cleared")  # old result cleared
    assert _by_id(out, "3").content == "x" * 400            # destructive result preserved (safety trail)
    assert _by_id(out, "4").content == "x" * 400            # most-recent kept (keep_recent=1)
    # every tool_use block (AI message with tool_calls) survives intact
    ai = [m for m in out if getattr(m, "type", "") == "ai"]
    assert len(ai) == 4 and all(m.tool_calls for m in ai)


def test_clearing_is_idempotent():
    msgs = _convo(400)
    once = clear_tool_results(msgs, threshold_tokens=50, keep_recent=1)
    twice = clear_tool_results(once, threshold_tokens=50, keep_recent=1)
    assert [getattr(m, "content", "") for m in once] == [getattr(m, "content", "") for m in twice]


class _FakeRequest:
    def __init__(self, messages):
        self.messages = messages

    def override(self, **overrides):
        return _FakeRequest(overrides.get("messages", self.messages))


def test_middleware_passes_cleared_messages_to_handler():
    from framework.context.clearing import ClearingMiddleware
    mw = ClearingMiddleware(threshold_tokens=50, keep_recent=1)
    received = {}

    def handler(req):
        received["messages"] = req.messages
        return "resp"

    out = mw.wrap_model_call(_FakeRequest(_convo(400)), handler)
    assert out == "resp"
    assert any(getattr(m, "content", "").startswith("[cleared") for m in received["messages"])
