"""External response tracing — a thin, backend-agnostic seam over OpenTelemetry.

Off by default. Turn on with `TRACE_BACKEND=otel`; spans then export to an OTLP collector when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set (e.g. a self-hosted Jaeger at http://localhost:4318), otherwise
to the console. The served RAG pipeline wraps each stage in `span(...)` so a request shows up as a
trace `rag_answer → condense → retrieve → generate → parse → verify` with per-stage latency and
attributes (n_hits, confidence, abstained, token counts, …).

Privacy: this corpus is sensitive, so raw TEXT payloads (question / answer / quote / condensed query)
are attached only when `TRACE_PAYLOADS=1`; by default spans carry structure + metrics only. Every
span op is wrapped so tracing can never break a response, and the whole module is a no-op when the
backend is off or the OTel SDK is unavailable.
"""

import json
import os
import warnings
from contextlib import contextmanager

# Attribute keys that may contain sensitive free text — redacted unless TRACE_PAYLOADS=1.
_TEXT_KEYS = {"question", "answer", "condensed_query", "quote"}

_state = {"ready": None, "tracer": None}       # ready: None=untried, True/False=outcome


def enabled() -> bool:
    return os.getenv("TRACE_BACKEND", "none").lower() == "otel"


def payloads_enabled() -> bool:
    return os.getenv("TRACE_PAYLOADS") == "1"


def _tracer():
    """Lazily build (once) the OTel tracer. Returns None if tracing is off or OTel is missing.
    If a TracerProvider is already set (e.g. a test's in-memory one), we reuse it."""
    if not enabled():
        return None
    if _state["ready"] is None:
        _state["ready"] = False
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "hrcp-rag")})
            provider = TracerProvider(resource=resource)
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            else:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)     # ignored if one is already set (that's fine)
            _state["tracer"] = trace.get_tracer("hrcp-rag")
            _state["ready"] = True
        except Exception as e:
            warnings.warn(f"tracing: OpenTelemetry unavailable, running without traces ({e})")
    return _state["tracer"] if _state["ready"] else None


def _coerce(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return json.dumps(value, default=str)[:4000]
    except Exception:
        return str(value)[:4000]


def _apply(span, attrs: dict) -> None:
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if k in _TEXT_KEYS and not payloads_enabled():
            continue                                # redact sensitive free text by default
        try:
            span.set_attribute(k, _coerce(v))
        except Exception:
            pass


class _Noop:
    def set_attribute(self, *a, **k):
        pass


_NOOP = _Noop()


@contextmanager
def span(name: str, **attrs):
    """Open a span `name` with `attrs` (no-op when tracing is off). Yields the span so the caller
    can add more attributes with `s.set_attribute(...)` or the `set_attrs(...)` helper.

    Yields EXACTLY once on every path. Only tracing SETUP is guarded (so a tracing failure can't
    break the request); an exception raised inside the caller's block propagates normally and is
    recorded on the span by OpenTelemetry."""
    tr = _tracer()
    if tr is None:
        yield _NOOP
        return
    try:
        cm = tr.start_as_current_span(name)
    except Exception:
        yield _NOOP
        return
    with cm as s:
        try:
            _apply(s, attrs)
        except Exception:
            pass
        yield s


def set_attrs(**attrs) -> None:
    """Add attributes to the currently-active span (no-op when tracing is off)."""
    if not enabled():
        return
    try:
        from opentelemetry import trace
        _apply(trace.get_current_span(), attrs)
    except Exception:
        pass
