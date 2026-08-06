"""Checkpointer factory — LangGraph persistence keyed by thread_id.

A checkpointer + a thread_id give an agent per-session memory: prior turns of a
thread are reloaded on the next call. "memory" is process-local; persistent
backends (sqlite/postgres) are a later, drop-in extension.
"""

from langgraph.checkpoint.memory import InMemorySaver


def checkpoint(kind: str = "memory", path: str | None = None):
    """Build a checkpointer. kind: 'memory' (default)."""
    if kind == "memory":
        return InMemorySaver()
    raise NotImplementedError(
        f"checkpoint kind {kind!r} not yet supported. Use 'memory'. "
        "Persistent backends (sqlite/postgres) are a planned extension."
    )
