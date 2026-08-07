from langchain.agents.middleware import ModelCallLimitMiddleware, HumanInTheLoopMiddleware
from framework.middleware import guardrails


def test_guardrails_builds_spend_cap_middleware():
    mw = guardrails(model_call_limit=3)
    assert any(isinstance(m, ModelCallLimitMiddleware) for m in mw)


def test_guardrails_builds_approval_gate_for_named_tools():
    mw = guardrails(require_approval=["delete_record"])
    assert any(isinstance(m, HumanInTheLoopMiddleware) for m in mw)


def test_guardrails_empty_by_default():
    assert guardrails() == []


def test_resilience_builds_retry_middleware():
    from langchain.agents.middleware import ModelRetryMiddleware
    from framework.middleware import resilience
    mw = resilience(max_retries=3)
    assert any(isinstance(m, ModelRetryMiddleware) for m in mw)
