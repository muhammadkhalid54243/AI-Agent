"""Tool-result clearing (mechanism 28) — the cheapest context lever.

Once a re-callable tool's result has been used, carrying its verbatim payload
forward costs attention on every later turn for nothing: if the agent needs it
again it can simply call the tool again. So we replace spent payloads with a
short placeholder.

Two rules are load-bearing:
  1. Keep the preceding tool_use block (the AI message's tool_calls) intact — the
     agent must still know it made the call, or a silent gap makes it re-run work
     it already did. We never touch AI messages here; only tool-result content.
  2. Never clear the most recent N results, results from a destructive tool
     (the safety trail must survive), or an already-cleared placeholder.

Clearing is mechanical — no LLM call — and idempotent. It runs before compaction
so we never pay an LLM to summarise payloads we were about to drop anyway.
"""

from langchain.agents.middleware import AgentMiddleware

from framework.context.budget import measure_messages

# Framework destructive stubs — their results are a safety record, never cleared.
DEFAULT_NON_CLEARABLE = frozenset({"send_email", "delete_record"})

_PREFIX = "[cleared"
# Neutral wording: state the body was omitted, do NOT invite a re-call. The
# preceding tool_use block already tells the agent it made the call; if it
# genuinely needs the value it can decide to call again on its own. Actively
# inviting a re-call causes an amnesia loop when the result was already spent.
_PLACEHOLDER = "[cleared: earlier {tool} result omitted to save context]"


def _is_tool_message(m) -> bool:
    if isinstance(m, dict):
        return m.get("role") == "tool"
    return getattr(m, "type", "") == "tool"


def _tool_name(m) -> str:
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    return name or "tool"


def _content(m) -> str:
    return (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""


def _replace_content(m, text):
    if isinstance(m, dict):
        nm = dict(m)
        nm["content"] = text
        return nm
    return m.model_copy(update={"content": text})  # preserves tool_call_id / name


def clear_tool_results(messages, *, threshold_tokens, keep_recent=3,
                       non_clearable=DEFAULT_NON_CLEARABLE):
    """Return messages with spent tool-result payloads replaced by placeholders.

    No-op (returns the same list object) when total tokens are within budget.
    """
    if measure_messages(messages)["total"] <= threshold_tokens:
        return messages

    tool_positions = [i for i, m in enumerate(messages) if _is_tool_message(m)]
    protected = set(tool_positions[-keep_recent:]) if keep_recent else set()

    out = []
    for i, m in enumerate(messages):
        if (_is_tool_message(m) and i not in protected
                and _tool_name(m) not in non_clearable
                and not _content(m).startswith(_PREFIX)):
            out.append(_replace_content(m, _PLACEHOLDER.format(tool=_tool_name(m))))
        else:
            out.append(m)
    return out


class ClearingMiddleware(AgentMiddleware):
    """Applies clear_tool_results to the request just before each model call.

    Edits only the outgoing request (via request.override) — the checkpointer's
    stored history keeps the full payloads, so nothing is permanently lost and
    the agent can re-call a tool for a value it needs again.
    """

    def __init__(self, *, threshold_tokens=1500, keep_recent=3, non_clearable=None):
        super().__init__()
        self._threshold = threshold_tokens
        self._keep = keep_recent
        self._non_clearable = (frozenset(non_clearable) if non_clearable is not None
                               else DEFAULT_NON_CLEARABLE)

    def wrap_model_call(self, request, handler):
        messages = getattr(request, "messages", None)
        if messages:
            cleared = clear_tool_results(
                messages, threshold_tokens=self._threshold,
                keep_recent=self._keep, non_clearable=self._non_clearable,
            )
            if cleared is not messages:
                request = request.override(messages=cleared)
        return handler(request)


def clearing(*, threshold_tokens=1500, keep_recent=3, non_clearable=None):
    """Middleware factory — compose with guardrails()/resilience() in an Agent."""
    return [ClearingMiddleware(threshold_tokens=threshold_tokens,
                               keep_recent=keep_recent, non_clearable=non_clearable)]
