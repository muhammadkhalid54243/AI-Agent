"""Response METRICS — a thin, backend-agnostic seam over OpenTelemetry metrics.

Jaeger (tracing.py) shows one request as a trace; this shows the AGGREGATE over time — how many
LLM calls, tokens burned, latency per stage, abstain rate, retrieval confidence — the numbers a
dashboard charts. Off by default; turn on with `METRICS_BACKEND=prometheus`, which starts a tiny
Prometheus endpoint on `METRICS_PORT` (default 9464, bound to 0.0.0.0 so a Prometheus in Docker can
scrape the host). Scrape it; point Grafana at Prometheus (see docker-compose.observability.yml).

Same discipline as tracing: zero overhead and a no-op when off or the OTel SDK is missing; every
record is wrapped so metrics can NEVER break a response; metrics are counts/latency only — no
question/answer/passage text ever leaves the process, so this is safe for the sensitive corpus.
"""

import os
import warnings

_state = {"ready": None, "inst": None}         # ready: None=untried, True/False=outcome


def enabled() -> bool:
    return os.getenv("METRICS_BACKEND", "none").lower() == "prometheus"


def _instruments():
    """Lazily build (once) the MeterProvider + Prometheus endpoint + instruments. Returns a dict of
    instruments, or None if metrics are off or OTel/prometheus_client is unavailable."""
    if not enabled():
        return None
    if _state["ready"] is None:
        _state["ready"] = False
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from prometheus_client import start_http_server

            reader = PrometheusMetricReader()
            resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "hrcp-rag")})
            provider = MeterProvider(metric_readers=[reader], resource=resource)
            start_http_server(int(os.getenv("METRICS_PORT", "9464")), addr="0.0.0.0")
            m = provider.get_meter("hrcp-rag")
            _state["inst"] = {
                "llm_calls": m.create_counter(
                    "hrcp_llm_calls", unit="1", description="LLM invocations, by pipeline stage"),
                "llm_tokens": m.create_counter(
                    "hrcp_llm_tokens", unit="1", description="LLM tokens, by stage and direction"),
                "requests": m.create_counter(
                    "hrcp_rag_requests", unit="1", description="RAG requests, by abstained flag"),
                # Explicit buckets: the OTel default boundaries ([0,5,10,25,…]) are meant for ms /
                # large counts and would collapse sub-second latencies and [0,1] confidences into the
                # first bucket, making every quantile meaningless. Size them to the real ranges.
                "latency": m.create_histogram(
                    "hrcp_stage_latency_seconds", unit="s", description="Per-stage wall-clock latency",
                    explicit_bucket_boundaries_advisory=[0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21]),
                "confidence": m.create_histogram(
                    "hrcp_retrieval_confidence", unit="1", description="Top-hit retrieval confidence",
                    explicit_bucket_boundaries_advisory=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6,
                                                         0.65, 0.7, 0.75, 0.8, 0.9]),
                "verifier": m.create_counter(
                    "hrcp_verifier_claims", unit="1", description="Verified claims, by outcome"),
            }
            _state["ready"] = True
        except Exception as e:                 # missing lib, port in use, anything — degrade to no-op
            warnings.warn(f"metrics: Prometheus/OTel unavailable, running without metrics ({e})")
    return _state["inst"] if _state["ready"] else None


def _safe(fn):
    """Run a record op, swallowing everything — metrics must never break a request."""
    inst = _instruments()
    if inst is None:
        return
    try:
        fn(inst)
    except Exception:
        pass


def llm_call(stage: str, input_tokens=None, output_tokens=None) -> None:
    def rec(i):
        i["llm_calls"].add(1, {"stage": stage})
        if input_tokens:
            i["llm_tokens"].add(int(input_tokens), {"stage": stage, "direction": "input"})
        if output_tokens:
            i["llm_tokens"].add(int(output_tokens), {"stage": stage, "direction": "output"})
    _safe(rec)


def request(abstained: bool) -> None:
    _safe(lambda i: i["requests"].add(1, {"abstained": str(bool(abstained)).lower()}))


def latency(stage: str, seconds: float) -> None:
    _safe(lambda i: i["latency"].record(max(0.0, float(seconds)), {"stage": stage}))


def confidence(value) -> None:
    if value is None:
        return
    _safe(lambda i: i["confidence"].record(float(value)))


def verifier_claims(kept: int = 0, dropped: int = 0) -> None:
    def rec(i):
        if kept:
            i["verifier"].add(int(kept), {"outcome": "kept"})
        if dropped:
            i["verifier"].add(int(dropped), {"outcome": "dropped"})
    _safe(rec)
