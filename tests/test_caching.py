from langchain_core.messages import SystemMessage, HumanMessage
from framework.context.caching import (
    stable_tool_signature, supports_caching, place_cache_breakpoints,
)


class _Tool:
    def __init__(self, name, description, args):
        self.name, self.description, self.args = name, description, args


def _tools():
    return [
        _Tool("get_weather", "Get weather", {"city": {"type": "string"}}),
        _Tool("calculate", "Do math", {"expression": {"type": "string"}}),
    ]


def test_tool_signature_is_order_independent():
    a = stable_tool_signature(_tools())
    b = stable_tool_signature(list(reversed(_tools())))
    assert a == b                       # reordering tools must not break the prefix
    assert isinstance(a, str) and len(a) > 0


def test_tool_signature_changes_when_a_tool_changes():
    base = stable_tool_signature(_tools())
    changed = _tools()
    changed[0].description = "Get the weather, now with wind"
    assert stable_tool_signature(changed) != base


def test_supports_caching_classifies_providers():
    assert supports_caching("anthropic:claude-x") == "explicit"
    assert supports_caching("groq:openai/gpt-oss-120b") == "automatic"
    assert supports_caching("openai:gpt-4o-mini") == "automatic"
    assert supports_caching("google_genai:gemini-2.0") == "implicit"


def test_place_breakpoints_noop_for_automatic_provider():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    assert place_cache_breakpoints(msgs, "groq:openai/gpt-oss-120b") is msgs


def test_place_breakpoints_marks_system_for_anthropic():
    msgs = [SystemMessage(content="big stable system prompt"), HumanMessage(content="hi")]
    out = place_cache_breakpoints(msgs, "anthropic:claude-x")
    # the system block carries a cache_control marker; the rest is untouched
    sys_block = out[0]
    assert isinstance(sys_block.content, list)
    assert any(part.get("cache_control") for part in sys_block.content if isinstance(part, dict))
