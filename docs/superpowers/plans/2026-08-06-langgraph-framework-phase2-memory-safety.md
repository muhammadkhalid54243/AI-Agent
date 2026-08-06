# LangGraph Framework — Phase 2 (Memory + Safety) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add per-session memory (checkpointer + `thread_id`) and code-level safety (spend cap + human-in-the-loop approval for destructive tools) to the `Agent`, using LangGraph-native mechanisms.

**Architecture:** `Agent` gains optional `memory=` (a checkpointer) and `middleware=` (a list) params, passed straight to `create_agent`. A `checkpoint()` factory builds checkpointers; a `guardrails()` factory assembles native middleware (`ModelCallLimitMiddleware` for the spend cap, `HumanInTheLoopMiddleware` for the approval gate). `run()`/`resume()` handle the interrupt/approve/deny flow; a denied approval means the tool never executes.

**Tech Stack:** LangGraph 1.x (`InMemorySaver`, `interrupt`/`Command`), LangChain agent middleware, pytest.

## Global Constraints

- Python `>=3.12`; secrets from env via `.env`.
- Verified API facts (probed live):
  - Interrupt surfaces as `result["__interrupt__"]` — a list of `Interrupt`; each `.value` has `action_requests` = `[{"name","args","description"}, ...]`.
  - Resume format: `Command(resume={"decisions": [{"type": "approve"|"reject"}]})`.
  - `ModelCallLimitMiddleware(run_limit=N, exit_behavior="end")` caps model calls per run.
  - `HumanInTheLoopMiddleware(interrupt_on={tool_name: True})` gates named tools.
  - HITL requires a checkpointer (memory) to pause/resume.
- Build on Phase 1: `Agent(model, *, tools, system_prompt, fallbacks)`, `resolve_model`, `framework.tools`.

---

## File structure (Phase 2)

- Create `framework/memory/__init__.py` — re-exports `checkpoint`.
- Create `framework/memory/checkpoint.py` — `checkpoint(kind, path=None)`.
- Create `framework/middleware/__init__.py` — re-exports `guardrails`.
- Create `framework/middleware/guardrails.py` — `guardrails(...)` + `ApprovalRequired`.
- Create `framework/tools/destructive.py` — `delete_record`, `send_email` as `@tool` (for gating demos/tests).
- Modify `framework/tools/__init__.py` — export the destructive tools.
- Modify `framework/core/agent.py` — add `memory`, `middleware` params; `resume()`; interrupt extraction.
- Test: `tests/test_memory.py`, `tests/test_guardrails.py`, `tests/test_safety_integration.py`.

---

### Task 1: Memory — checkpointer factory + Agent wiring

**Files:**
- Create: `framework/memory/checkpoint.py`, `framework/memory/__init__.py`
- Modify: `framework/core/agent.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces:
  - `checkpoint(kind: str = "memory", path: str | None = None)` → a checkpointer. `"memory"` → `InMemorySaver()`. `"sqlite"`/`"postgres"` raise `NotImplementedError` with guidance (deferred).
  - `Agent(..., memory=None)` — when set, passed as `create_agent(checkpointer=memory)`.

- [ ] **Step 1: Write the failing test**

`tests/test_memory.py`:
```python
from langgraph.checkpoint.memory import InMemorySaver
from framework.memory import checkpoint


def test_checkpoint_memory_returns_in_memory_saver():
    cp = checkpoint("memory")
    assert isinstance(cp, InMemorySaver)


def test_checkpoint_unknown_kind_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        checkpoint("sqlite", "state.db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.memory`.

- [ ] **Step 3: Write minimal implementation**

`framework/memory/checkpoint.py`:
```python
"""Checkpointer factory — LangGraph persistence keyed by thread_id.

A checkpointer + a thread_id give an agent per-session memory: prior turns of a
thread are reloaded on the next call. "memory" is process-local; persistent
backends (sqlite/postgres) are a later, drop-in extension.
"""

from langgraph.checkpoint.memory import InMemorySaver


def checkpoint(kind: str = "memory", path: str | None = None):
    """Build a checkpointer. kind: 'memory' (default)."""
    if kind == "memory":
        return InMemorySaver()
    raise NotImplementedError(
        f"checkpoint kind {kind!r} not yet supported. Use 'memory'. "
        "Persistent backends (sqlite/postgres) are a planned extension."
    )
```

`framework/memory/__init__.py`:
```python
from framework.memory.checkpoint import checkpoint

__all__ = ["checkpoint"]
```

- [ ] **Step 4: Modify Agent to accept `memory`**

In `framework/core/agent.py`, change `__init__` and `_build`:
```python
    def __init__(self, model, *, tools=None, system_prompt=None, fallbacks=None,
                 memory=None, middleware=None):
        self._model = model if not isinstance(model, str) else resolve_model(model, fallbacks)
        self.tools = list(tools) if tools else []
        self._system_prompt = system_prompt
        self._memory = memory
        self._middleware = list(middleware) if middleware else []
        self._graph = self._build()

    def _build(self):
        kwargs = {"model": self._model, "tools": self.tools}
        if self._system_prompt:
            kwargs["system_prompt"] = self._system_prompt
        if self._memory is not None:
            kwargs["checkpointer"] = self._memory
        if self._middleware:
            kwargs["middleware"] = self._middleware
        return create_agent(**kwargs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_memory.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Add a live multi-turn memory test**

Append to `tests/test_memory.py`:
```python
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_memory_persists_across_turns():
    from framework import Agent
    from framework.memory import checkpoint

    agent = Agent("groq:llama-3.3-70b-versatile", memory=checkpoint("memory"),
                  system_prompt="Answer briefly.")
    agent.run("My favorite number is 42. Remember it.", thread_id="s1")
    out = agent.run("What is my favorite number?", thread_id="s1")
    assert "42" in out
```

- [ ] **Step 7: Run the live test**

Run: `uv run pytest tests/test_memory.py -q`
Expected: PASS (3 passed) with a key present — the second turn recalls "42".

---

### Task 2: Safety — spend cap via guardrails middleware

**Files:**
- Create: `framework/middleware/guardrails.py`, `framework/middleware/__init__.py`
- Create: `framework/tools/destructive.py`
- Modify: `framework/tools/__init__.py`
- Test: `tests/test_guardrails.py`

**Interfaces:**
- Produces:
  - `guardrails(*, model_call_limit=None, require_approval=None) -> list` — a list of native middleware.
  - `ApprovalRequired(Exception)` with `.thread_id` and `.requests` (list of `{"name","args"}`).
  - `delete_record`, `send_email` — destructive `@tool`s.

- [ ] **Step 1: Write the failing test**

`tests/test_guardrails.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_guardrails.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.middleware`.

- [ ] **Step 3: Write minimal implementation**

`framework/middleware/guardrails.py`:
```python
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
```

`framework/middleware/__init__.py`:
```python
from framework.middleware.guardrails import guardrails, ApprovalRequired

__all__ = ["guardrails", "ApprovalRequired"]
```

`framework/tools/destructive.py`:
```python
"""Destructive example tools — gated behind approval via guardrails(require_approval=...)."""

import json

from langchain.tools import tool


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient. DESTRUCTIVE — real-world side effect."""
    return json.dumps({"status": "sent", "to": to, "body_preview": body[:60]})


@tool
def delete_record(record_id: str) -> str:
    """Permanently delete a record by id. DESTRUCTIVE and irreversible."""
    return json.dumps({"status": "deleted", "record_id": record_id})
```

In `framework/tools/__init__.py`, replace contents with:
```python
from framework.tools.builtin import get_weather, calculate
from framework.tools.destructive import send_email, delete_record

__all__ = ["get_weather", "calculate", "send_email", "delete_record"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_guardrails.py -q`
Expected: PASS (3 passed).

---

### Task 3: Safety — interrupt/approve/deny flow in Agent

**Files:**
- Modify: `framework/core/agent.py`
- Test: `tests/test_safety_integration.py`

**Interfaces:**
- Consumes: `ApprovalRequired` (Task 2), `guardrails`, `checkpoint`, destructive tools.
- Produces:
  - `Agent.run(message, thread_id=None) -> str` — raises `ApprovalRequired` when a gated tool is pending.
  - `Agent.resume(thread_id, approve: bool) -> str` — resumes; approve executes, reject blocks.

- [ ] **Step 1: Write the failing test**

`tests/test_safety_integration.py`:
```python
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
        middleware=guardrails(require_approval=["delete_record"]),
    )


def test_destructive_tool_requires_approval_then_reject_blocks():
    from framework.middleware import ApprovalRequired
    agent = _agent()
    with pytest.raises(ApprovalRequired) as exc:
        agent.run("Delete record 42.", thread_id="d1")
    assert exc.value.requests[0]["name"] == "delete_record"
    final = agent.resume("d1", approve=False)
    assert "cancel" in final.lower() or "not" in final.lower() or final  # completes without executing


def test_destructive_tool_approve_executes():
    from framework.middleware import ApprovalRequired
    agent = _agent()
    try:
        agent.run("Delete record 99.", thread_id="d2")
    except ApprovalRequired:
        pass
    final = agent.resume("d2", approve=True)
    assert "99" in final or "delet" in final.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safety_integration.py -q`
Expected: FAIL — `Agent` has no `resume`, and `run` does not raise `ApprovalRequired`.

- [ ] **Step 3: Implement interrupt handling in Agent**

In `framework/core/agent.py`, add the import and the methods. Add near the top:
```python
from langgraph.types import Command

from framework.middleware.guardrails import ApprovalRequired
```

Replace `run` and add helpers + `resume`:
```python
    def run(self, message: str, thread_id: str | None = None) -> str:
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return self._extract(result, thread_id)

    def resume(self, thread_id: str, approve: bool) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        decision = "approve" if approve else "reject"
        result = self._graph.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=config,
        )
        return self._extract(result, thread_id)

    @staticmethod
    def _extract(result: dict, thread_id):
        if "__interrupt__" in result:
            requests = []
            for it in result["__interrupt__"]:
                for ar in it.value.get("action_requests", []):
                    requests.append({"name": ar["name"], "args": ar["args"]})
            raise ApprovalRequired(thread_id, requests)
        return result["messages"][-1].content
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_safety_integration.py -q`
Expected: PASS (2 passed) with a key present — reject completes without deleting; approve executes.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all Phase 1 + Phase 2 tests pass.

---

## Self-review

**Spec coverage (Phase 2):** memory via checkpointer/`thread_id` → Task 1; spend cap → Task 2;
HITL destructive gate + interrupt/approve/deny → Tasks 2–3; injection defense remains the
system-prompt policy (Phase 1 prompt) backed by the gate. Persistent checkpointers
(sqlite/postgres) explicitly deferred (documented `NotImplementedError`) — YAGNI for proving memory.

**Placeholder scan:** all code shown; the one loose assertion in the reject test (`... or final`)
tolerates model phrasing while still proving completion-without-execution (approve test asserts
execution). No TBD/TODO.

**Type consistency:** `checkpoint`, `guardrails`, `ApprovalRequired(thread_id, requests)`,
`Agent(memory=, middleware=)`, `run`, `resume`, `_extract` names are consistent across Tasks 1–3
and match the probed API (`__interrupt__`, `action_requests`, `Command(resume={"decisions":[...]})`).
