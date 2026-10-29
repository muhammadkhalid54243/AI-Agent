"""Shared test fixtures."""

import pytest

from framework.observability import tracing


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Keep response tracing OFF during tests by default. A developer's `.env` may set
    TRACE_BACKEND=otel (to view traces in Jaeger), and serve's load_dotenv() would otherwise leak
    that into the test process — turning tracing on everywhere and shipping test spans to a
    collector. Tests that exercise tracing (test_tracing.py) re-enable it per-test."""
    monkeypatch.setenv("TRACE_BACKEND", "none")
    tracing._state["ready"] = None
    tracing._state["tracer"] = None
    yield
