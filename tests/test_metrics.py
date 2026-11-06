"""Response metrics seam (offline) — off by default, records + exports when on."""

import importlib

from framework.observability import metrics


def _reset():
    metrics._state["ready"] = None
    metrics._state["inst"] = None


def test_disabled_by_default_is_a_noop(monkeypatch):
    monkeypatch.delenv("METRICS_BACKEND", raising=False)
    _reset()
    assert metrics.enabled() is False
    # None of these must raise, and nothing is recorded.
    metrics.llm_call("generate", 10, 20)
    metrics.request(abstained=False)
    metrics.latency("generate", 0.1)
    metrics.confidence(0.7)
    metrics.verifier_claims(kept=2, dropped=1)
    assert metrics._instruments() is None


def test_enabled_records_and_exports(monkeypatch):
    monkeypatch.setenv("METRICS_BACKEND", "prometheus")
    monkeypatch.setenv("METRICS_PORT", "0")                 # ephemeral port — don't clash with 9464
    _reset()
    assert metrics.enabled() is True
    metrics.llm_call("generate", 12, 34)
    metrics.request(abstained=False)
    metrics.latency("retrieve", 0.05)
    metrics.confidence(0.63)
    metrics.verifier_claims(kept=3, dropped=1)

    from prometheus_client import REGISTRY, generate_latest
    out = generate_latest(REGISTRY).decode()
    assert "hrcp_llm_calls" in out                          # counter exported
    assert "hrcp_rag_requests" in out
    assert "hrcp_stage_latency_seconds" in out              # histogram exported
    assert "hrcp_retrieval_confidence" in out
    _reset()
