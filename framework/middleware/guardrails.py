"""Guardrails — assemble native LangChain agent middleware for safety.

Two controls, both enforced in the execution layer (not the prompt):
  - spend cap: ModelCallLimitMiddleware ends the run after N model calls.
  - approval gate: HumanInTheLoopMiddleware pauses (interrupt) before a named
    destructive tool runs; a denied approval means the tool body never executes.
"""

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    HumanInTheLoopMiddleware,
)


class ApprovalRequired(Exception):
    """Raised by Agent.run/resume when a destructive tool awaits human approval."""

    def __init__(self, thread_id, requests):
        self.thread_id = thread_id
        self.requests = requests  # list of {"name": str, "args": dict}
        names = ", ".join(r["name"] for r in requests)
        super().__init__(f"Approval required for: {names}")


def guardrails(*, model_call_limit=None, require_approval=None):
    """Build the middleware list from a simple config."""
    mw = []
    if model_call_limit is not None:
        mw.append(ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"))
    if require_approval:
        mw.append(HumanInTheLoopMiddleware(interrupt_on={name: True for name in require_approval}))
    return mw
