"""Resilience — automatic retry of transient model failures.

Providers occasionally return transient errors (e.g. a malformed tool-call 400).
resilience() wraps model calls in retry-with-backoff so those self-heal instead
of surfacing to the caller. Combine with guardrails() in an Agent's middleware
list. Cross-provider fallback lives separately on the model (resolve_model
fallbacks); this is same-provider retry.
"""

from langchain.agents.middleware import ModelRetryMiddleware


def resilience(*, max_retries: int = 2, backoff_factor: float = 2.0):
    """Return middleware that retries failed model calls with exponential backoff."""
    return [ModelRetryMiddleware(max_retries=max_retries, backoff_factor=backoff_factor)]
