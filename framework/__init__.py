"""framework — a modular LangGraph-based agent framework."""

__all__ = ["Agent"]


def __getattr__(name):
    # Lazy top-level export: `from framework import Agent` works without eagerly
    # importing the whole subtree (and survives partial builds during development).
    if name == "Agent":
        from framework.core.agent import Agent
        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
