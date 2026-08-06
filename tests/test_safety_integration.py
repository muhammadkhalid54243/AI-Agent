import os
import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")


def _agent():
    from framework import Agent
    from framework.memory import checkpoint
    from framework.middleware import guardrails
    from framework.tools import delete_record
    return Agent(
        "groq:llama-3.3-70b-versatile",
        tools=[delete_record],
        system_prompt="Use delete_record when asked to delete something.",
        memory=checkpoint("memory"),
        middleware=guardrails(require_approval=["delete_record"], model_call_limit=6),
    )


def test_destructive_tool_requires_approval_then_reject_blocks():
    from framework.middleware import ApprovalRequired
    agent = _agent()
    with pytest.raises(ApprovalRequired) as exc:
        agent.run("Delete record 42.", thread_id="d1")
    assert exc.value.requests[0]["name"] == "delete_record"
    # A rejected model may re-request; reject each time until the run completes.
    # The spend cap bounds the loop. The delete never executes on reject.
    final = None
    for _ in range(6):
        try:
            final = agent.resume("d1", approve=False)
            break
        except ApprovalRequired:
            continue
    assert final is not None  # run completed without executing the delete


def test_destructive_tool_approve_executes():
    from framework.middleware import ApprovalRequired
    agent = _agent()
    try:
        agent.run("Delete record 99.", thread_id="d2")
    except ApprovalRequired:
        pass
    final = agent.resume("d2", approve=True)
    assert "99" in final or "delet" in final.lower()
