"""Prompt caching (mechanism 11) — stop re-paying for a large static prefix.

Caching is worthless until the message pipeline is stable (every earlier context
edit invalidates a breakpoint), which is why this is the last of the shrink/keep
mechanisms. Economics: a cache WRITE costs ~1.25x base input, a READ ~0.1x — so
caching loses on a single turn and wins big from turn two, which is the whole
reason "put the document in context" is a real strategy.

Providers differ:
  - explicit  (Anthropic): you place `cache_control` markers on the stable prefix.
  - automatic (OpenAI/Groq/OpenRouter): cached by exact prefix match, no markers —
    the framework's only job is to keep that prefix byte-stable.
  - implicit  (Gemini): provider-managed.

Two levers here regardless of provider:
  1. `stable_tool_signature` — deterministic tool serialisation. Reordered tools or
     shuffled JSON keys silently break an automatic cache; a stable signature does not.
  2. `place_cache_breakpoints` — for explicit providers, mark the stable prefix.
"""

import json

_EXPLICIT = {"anthropic"}
_AUTOMATIC = {"groq", "openai", "openrouter", "openroute"}
_IMPLICIT = {"google", "google_genai", "gemini"}


def _provider(model_spec: str) -> str:
    return str(model_spec).split(":", 1)[0].lower()


def supports_caching(model_spec: str) -> str:
    """Classify how a provider caches: 'explicit' | 'automatic' | 'implicit' | 'none'."""
    p = _provider(model_spec)
    if p in _EXPLICIT:
        return "explicit"
    if p in _AUTOMATIC:
        return "automatic"
    if p in _IMPLICIT:
        return "implicit"
    return "none"


def stable_tool_signature(tools) -> str:
    """A deterministic, order-independent serialisation of a tool set.

    Sorted by name, canonical JSON (sorted keys, fixed separators) so the tool
    portion of the prefix is byte-identical across runs — a prerequisite for an
    automatic cache hit.
    """
    items = []
    for t in tools or []:
        items.append({
            "name": getattr(t, "name", ""),
            "description": getattr(t, "description", ""),
            "args": getattr(t, "args", None),
        })
    items.sort(key=lambda d: d["name"])
    return json.dumps(items, sort_keys=True, separators=(",", ":"), default=str)


def place_cache_breakpoints(messages, model_spec: str):
    """For explicit-cache providers, mark the stable prefix (the system message).

    No-op (returns the same list) for automatic/implicit/none providers, where
    markers are meaningless and prefix stability alone drives the cache.
    """
    if supports_caching(model_spec) != "explicit":
        return messages
    if not messages:
        return messages

    first = messages[0]
    content = getattr(first, "content", None)
    if not isinstance(content, str):
        return messages  # already structured; leave it

    marked = first.model_copy(update={
        "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    })
    return [marked] + list(messages[1:])
