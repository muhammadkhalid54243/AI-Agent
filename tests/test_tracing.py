"""Response tracing seam — offline, no network (OTel InMemory exporter)."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from framework.observability import tracing

# Set a real provider ONCE for the whole test session (first setter wins in OTel); each test
# clears the exporter. tracing._tracer() reuses this global provider, so spans land here.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setenv("TRACE_BACKEND", "otel")
    tracing._state["ready"] = None
    tracing._state["tracer"] = None
    _EXPORTER.clear()
    yield _EXPORTER
    tracing._state["ready"] = None
    tracing._state["tracer"] = None


def test_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("TRACE_BACKEND", raising=False)
    tracing._state["ready"] = None
    tracing._state["tracer"] = None
    with tracing.span("x", a=1) as s:
        s.set_attribute("b", 2)                 # yields a no-op span; must not error
    tracing.set_attrs(c=3)                       # no current span; must not error
    assert tracing.enabled() is False


def test_graceful_when_tracer_unavailable(monkeypatch):
    monkeypatch.setenv("TRACE_BACKEND", "otel")
    tracing._state["ready"] = False              # simulate OTel init having failed
    tracing._state["tracer"] = None
    with tracing.span("x", a=1) as s:
        s.set_attribute("b", 2)                  # still a no-op, no crash


def test_span_tree_and_attributes_recorded(recorder):
    with tracing.span("rag_answer"):
        with tracing.span("retrieve"):
            tracing.set_attrs(n_hits=5, confidence=0.73, abstained=False)
        with tracing.span("generate"):
            tracing.set_attrs(input_tokens=1200, output_tokens=80)
    names = [s.name for s in recorder.get_finished_spans()]
    assert {"rag_answer", "retrieve", "generate"} <= set(names)
    retrieve = next(s for s in recorder.get_finished_spans() if s.name == "retrieve")
    assert retrieve.attributes.get("n_hits") == 5
    assert retrieve.attributes.get("confidence") == pytest.approx(0.73)


def test_complex_attrs_are_json_encoded(recorder):
    with tracing.span("retrieve", hits=[{"doc": "SHR 2024", "chapter": "Balochistan", "page": 176}]):
        pass
    s = recorder.get_finished_spans()[0]
    assert "Balochistan" in s.attributes.get("hits")      # list→JSON string


def test_text_payload_redacted_by_default(recorder, monkeypatch):
    monkeypatch.delenv("TRACE_PAYLOADS", raising=False)
    with tracing.span("rag_answer", question="a sensitive question"):
        pass
    s = recorder.get_finished_spans()[0]
    assert "question" not in (s.attributes or {})


def test_text_payload_present_when_enabled(recorder, monkeypatch):
    monkeypatch.setenv("TRACE_PAYLOADS", "1")
    with tracing.span("rag_answer", question="a sensitive question"):
        pass
    s = recorder.get_finished_spans()[0]
    assert s.attributes.get("question") == "a sensitive question"
