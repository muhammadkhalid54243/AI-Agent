"""Token accounting — the measure-first foundation for the context layer.

Every later mechanism (clearing, compaction, caching) is a cost/quality tradeoff,
and a tradeoff you cannot measure you cannot evaluate. This module counts tokens
per message *section* so we can see where the budget actually goes — the plan's
prediction is that spent tool results dominate.

Tokenization note: providers (Groq/Llama, Gemini) don't publish exact tokenizers,
so we use tiktoken's cl100k_base as a stable approximation for attribution. The
authoritative totals still come from each provider's reported usage (captured by
the tracer); this gives the per-section breakdown the provider doesn't.
"""

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# role → section. Anything unknown falls into "history".
_SECTION_BY_ROLE = {
    "system": "system",
    "tool": "tool_results",
    "user": "history",
    "human": "history",
    "assistant": "history",
    "ai": "history",
}


def count_tokens(value) -> int:
    """Approximate token count for a string (or the str() of any value)."""
    if value is None:
        return 0
    if not isinstance(value, str):
        value = str(value)
    if not value:
        return 0
    return len(_ENC.encode(value))


def _role(message) -> str:
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "type", "")  # BaseMessage: system/human/ai/tool


def _content(message) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def _tool_call_tokens(message) -> int:
    """Tokens for tool-call requests on an assistant message (name + arguments)."""
    calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    if not calls:
        return 0
    return count_tokens(calls)


def measure_messages(messages) -> dict:
    """Return per-section token counts for a message list, plus a total."""
    sections = {"system": 0, "history": 0, "tool_results": 0}
    for m in messages:
        section = _SECTION_BY_ROLE.get(_role(m), "history")
        sections[section] += count_tokens(_content(m))
        if section == "history":
            sections["history"] += _tool_call_tokens(m)
    sections["total"] = sections["system"] + sections["history"] + sections["tool_results"]
    return sections


def measure_tools(tools) -> int:
    """Approximate token cost of the tool definitions bound to the model."""
    total = 0
    for t in tools or []:
        total += count_tokens(getattr(t, "name", ""))
        total += count_tokens(getattr(t, "description", ""))
        total += count_tokens(getattr(t, "args", None))
    return total


class TokenBudget:
    """Accumulates per-turn, per-section token measurements across a run."""

    def __init__(self):
        self.turns: list[dict] = []

    def record(self, messages, tools=None) -> dict:
        turn = measure_messages(messages)
        if tools:
            turn["tools"] = measure_tools(tools)
            turn["total"] += turn["tools"]
        self.turns.append(turn)
        return turn

    def peak_total(self) -> int:
        return max((t["total"] for t in self.turns), default=0)

    def report(self) -> dict:
        """Aggregate view: peak context, and total tokens summed per section."""
        agg = {"system": 0, "history": 0, "tool_results": 0, "tools": 0}
        for t in self.turns:
            for k in agg:
                agg[k] += t.get(k, 0)
        return {
            "turns": len(self.turns),
            "peak_total": self.peak_total(),
            "by_section_summed": agg,
        }
