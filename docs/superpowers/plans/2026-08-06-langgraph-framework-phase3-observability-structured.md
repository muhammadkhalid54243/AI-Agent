# LangGraph Framework — Phase 3 (Observability + Structured Output) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add trajectory observability (a callback handler that records model/tool calls with timing) and structured output (native `with_structured_output` helper) to the framework.

**Architecture:** `TrajectoryTracer` is a LangChain `BaseCallbackHandler` recording timed events; `Agent.run_traced()` attaches it and returns `{answer, summary, events}`. `extract()` wraps a model's `with_structured_output` to return a validated Pydantic instance (or dict) from free text.

**Tech Stack:** LangChain callbacks (`BaseCallbackHandler`), `with_structured_output`, Pydantic, pytest.

## Global Constraints

- Python `>=3.12`; secrets from env via `.env`.
- Verified API facts (probed live):
  - A callback attached via `config={"callbacks": [handler]}` fires `on_chat_model_start`/`on_llm_start`, `on_tool_start(name=...)`, `on_tool_end`, `on_llm_end` during a graph invoke.
  - `model.with_structured_output(PydanticModel).invoke(text)` returns a validated model instance.
- Build on Phases 1–2: `Agent`, `_extract` (raises `ApprovalRequired` on interrupt), `resolve_model`.

---

## File structure (Phase 3)

- Create `framework/observability/__init__.py` — re-exports `TrajectoryTracer`.
- Create `framework/observability/tracer.py` — `TrajectoryTracer`.
- Create `framework/structured/__init__.py` — re-exports `extract`.
- Create `framework/structured/extract.py` — `extract(model, text, schema, instruction=None)`.
- Modify `framework/core/agent.py` — add `run_traced()`.
- Test: `tests/test_tracer.py`, `tests/test_structured.py`.

---

### Task 1: Observability — TrajectoryTracer + Agent.run_traced

**Files:**
- Create: `framework/observability/tracer.py`, `framework/observability/__init__.py`
- Modify: `framework/core/agent.py`
- Test: `tests/test_tracer.py`

**Interfaces:**
- Produces:
  - `TrajectoryTracer()` — a `BaseCallbackHandler` with `.events` (list of `{kind, duration_ms, [tool]}`) and `.summary() -> {model_calls, tool_calls, total_ms}`.
  - `Agent.run_traced(message, thread_id=None) -> dict` — `{"answer", "summary", "events"}`; raises `ApprovalRequired` on interrupt.

- [ ] **Step 1: Write the failing unit test**

`tests/test_tracer.py`:
```python
from framework.observability import TrajectoryTracer


def test_tracer_counts_model_and_tool_calls():
    t = TrajectoryTracer()
    t.on_chat_model_start({}, [], run_id="a")
    t.on_llm_end(None, run_id="a")
    t.on_tool_start({}, "", run_id="b", name="get_weather")
    t.on_tool_end("", run_id="b")
    s = t.summary()
    assert s["model_calls"] == 1
    assert s["tool_calls"] == 1
    assert s["total_ms"] >= 0
    assert any(e.get("tool") == "get_weather" for e in t.events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracer.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.observability`.

- [ ] **Step 3: Write the tracer**

`framework/observability/tracer.py`:
```python
"""TrajectoryTracer — a callback handler that records what the agent did.

Attached via config={"callbacks": [tracer]}, it times each model call and tool
call by matching start/end on run_id, then summarizes the trajectory. This is the
observability layer: see model_calls, tool_calls, and total latency after a run.
"""

import time

from langchain_core.callbacks import BaseCallbackHandler


class TrajectoryTracer(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self._starts = {}  # run_id -> (kind, start_perf, meta)

    # chat models call on_chat_model_start; completion models call on_llm_start
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self._starts[run_id] = ("model_call", time.perf_counter(), {})

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._starts[run_id] = ("model_call", time.perf_counter(), {})

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        self._finish(run_id)

    def on_tool_start(self, serialized, input_str, *, run_id=None, name=None, **kwargs):
        tool_name = name or (serialized or {}).get("name", "tool")
        self._starts[run_id] = ("tool_call", time.perf_counter(), {"tool": tool_name})

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        self._finish(run_id)

    def _finish(self, run_id):
        entry = self._starts.pop(run_id, None)
        if not entry:
            return
        kind, start, meta = entry
        self.events.append({
            "kind": kind,
            "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            **meta,
        })

    def summary(self) -> dict:
        model_calls = sum(1 for e in self.events if e["kind"] == "model_call")
        tool_calls = sum(1 for e in self.events if e["kind"] == "tool_call")
        total_ms = round(sum(e.get("duration_ms", 0) for e in self.events), 1)
        return {"model_calls": model_calls, "tool_calls": tool_calls, "total_ms": total_ms}
```

`framework/observability/__init__.py`:
```python
from framework.observability.tracer import TrajectoryTracer

__all__ = ["TrajectoryTracer"]
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `uv run pytest tests/test_tracer.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Add `run_traced` to Agent**

In `framework/core/agent.py`, add the import near the others:
```python
from framework.observability.tracer import TrajectoryTracer
```

Add this method after `run`:
```python
    def run_traced(self, message: str, thread_id: str | None = None) -> dict:
        """Run and return {answer, summary, events} with a full trajectory trace."""
        tracer = TrajectoryTracer()
        config = {"callbacks": [tracer]}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        answer = self._extract(result, thread_id)  # raises ApprovalRequired on interrupt
        return {"answer": answer, "summary": tracer.summary(), "events": tracer.events}
```

- [ ] **Step 6: Add a live trajectory test**

Append to `tests/test_tracer.py`:
```python
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_run_traced_reports_trajectory():
    from framework import Agent
    from framework.tools import get_weather

    agent = Agent("groq:llama-3.3-70b-versatile", tools=[get_weather],
                  system_prompt="Use tools to answer.")
    result = agent.run_traced("What is the weather in Lahore?")
    assert "42" in result["answer"]
    assert result["summary"]["model_calls"] >= 1
    assert result["summary"]["tool_calls"] >= 1
```

- [ ] **Step 7: Run the live test**

Run: `uv run pytest tests/test_tracer.py -q`
Expected: PASS (2 passed) with a key present.

---

### Task 2: Structured output — extract()

**Files:**
- Create: `framework/structured/extract.py`, `framework/structured/__init__.py`
- Test: `tests/test_structured.py`

**Interfaces:**
- Produces: `extract(model, text, schema, instruction=None)` — `model` is a spec string or a chat model; `schema` is a Pydantic model class or a JSON-schema dict; returns a validated instance (or dict).

- [ ] **Step 1: Write the failing test (live)**

`tests/test_structured.py`:
```python
import os
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")


class Ticket(BaseModel):
    summary: str
    priority: str


def test_extract_returns_validated_model():
    from framework.structured import extract
    ticket = extract(
        "groq:llama-3.3-70b-versatile",
        "Login returns 500 for all users since this morning. Urgent.",
        Ticket,
        instruction="Turn this into a support ticket.",
    )
    assert isinstance(ticket, Ticket)
    assert ticket.priority  # non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_structured.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.structured`.

- [ ] **Step 3: Write the implementation**

`framework/structured/extract.py`:
```python
"""Structured output — force a model to answer as a validated schema.

Uses LangChain's native with_structured_output (tool-calling / JSON mode under
the hood), so parsing and validation are handled by the provider integration
instead of hand-rolled JSON repair. Pass a Pydantic model for typed, validated
results, or a JSON-schema dict for a plain dict.
"""

from framework.core.models import resolve_model


def extract(model, text: str, schema, instruction: str | None = None):
    """Return `text` coerced into `schema`. `model` may be a spec string or a chat model."""
    m = model if not isinstance(model, str) else resolve_model(model)
    structured = m.with_structured_output(schema)
    content = f"{instruction}\n\n{text}" if instruction else text
    return structured.invoke(content)
```

`framework/structured/__init__.py`:
```python
from framework.structured.extract import extract

__all__ = ["extract"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_structured.py -q`
Expected: PASS (1 passed) with a key present.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all Phase 1–3 tests pass.

---

## Self-review

**Spec coverage (Phase 3):** observability via callback tracer + `run_traced` → Task 1;
structured output via native `with_structured_output` → Task 2. Both replace the plain-Python
hand-rolled versions (`TrajectoryLogger`, `extract_json`) with native mechanisms.

**Placeholder scan:** all code shown; the live tests are network-gated with clear skip reasons;
no TBD/TODO.

**Type consistency:** `TrajectoryTracer` (`events`, `summary`), `Agent.run_traced -> {answer,
summary, events}`, and `extract(model, text, schema, instruction)` are consistent across tasks and
match the probed callback and `with_structured_output` APIs.
