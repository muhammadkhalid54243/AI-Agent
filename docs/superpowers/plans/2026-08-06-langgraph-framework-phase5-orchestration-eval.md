# LangGraph Framework — Phase 5 (Orchestration + Eval) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add orchestration (sub-agents-as-tools + a `supervisor()` builder) and an evaluation harness that scores a composed `Agent` on outcome + trajectory + LLM-judge.

**Architecture:** `subagent(agent, name, description)` wraps an `Agent` as a delegable tool via `StructuredTool.from_function`; `supervisor(model, subagents, ...)` builds an orchestrator `Agent` that delegates. The eval `runner` drives the real `Agent.run_traced`, reading the trajectory from its events; the `judge` reuses `structured.extract` for a validated score.

**Tech Stack:** `StructuredTool`, the framework's own `Agent`, `structured.extract`, pytest.

## Global Constraints

- Python `>=3.12`; secrets from env via `.env`.
- Verified API facts (probed live):
  - `StructuredTool.from_function(func=<callable(question)>, name=..., description=...)` → a tool; `.invoke({"question": ...})` runs the callable.
- Build on Phases 1–4: `Agent` (with `run`, `run_traced`), `structured.extract`, `framework.tools`.

---

## File structure (Phase 5)

- Create `framework/orchestration/__init__.py` — re-exports `subagent`, `supervisor`.
- Create `framework/orchestration/supervisor.py` — `subagent()`, `supervisor()`.
- Create `framework/eval/__init__.py` — re-exports `run_suite`, `score_case`, `EVAL_SET`.
- Create `framework/eval/dataset.py` — `EVAL_SET`.
- Create `framework/eval/judge.py` — `judge_answer()`.
- Create `framework/eval/runner.py` — `score_case()`, `run_suite()`.
- Test: `tests/test_orchestration.py`, `tests/test_eval.py`.

---

### Task 1: Orchestration — subagent() + supervisor()

**Files:**
- Create: `framework/orchestration/supervisor.py`, `framework/orchestration/__init__.py`
- Test: `tests/test_orchestration.py`

**Interfaces:**
- Produces:
  - `subagent(agent, name, description) -> BaseTool` — a tool that forwards `question` to `agent.run`.
  - `supervisor(model, subagents, *, system_prompt=None, **kwargs) -> Agent`.

- [ ] **Step 1: Write the failing test (no network — fake agent)**

`tests/test_orchestration.py`:
```python
from framework.orchestration import subagent


class FakeAgent:
    def run(self, question, thread_id=None):
        return f"handled: {question}"


def test_subagent_wraps_agent_as_named_tool():
    tool = subagent(FakeAgent(), "ask_researcher", "Delegate research questions.")
    assert tool.name == "ask_researcher"
    assert tool.invoke({"question": "what is RAG?"}) == "handled: what is RAG?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.orchestration`.

- [ ] **Step 3: Write the implementation**

`framework/orchestration/supervisor.py`:
```python
"""Orchestration — delegate to specialist sub-agents.

A sub-agent is just an Agent wrapped as a tool: the supervisor calls it with a
question and gets back its answer. The supervisor is itself an Agent whose tools
are those sub-agents, so it plans, delegates, and synthesizes with the same loop.
Use multiple agents only when the task needs distinct expertise — a single
well-tooled agent is often better.
"""

from langchain_core.tools import StructuredTool

from framework.core.agent import Agent

_DEFAULT_SUPERVISOR_PROMPT = (
    "You are an orchestrator. Break the task down and delegate each part to the "
    "appropriate specialist tool. Then synthesize one clear final answer."
)


def subagent(agent, name: str, description: str):
    """Wrap an Agent as a delegable tool. The tool takes a single `question` string."""
    def _call(question: str) -> str:
        return agent.run(question)

    return StructuredTool.from_function(func=_call, name=name, description=description)


def supervisor(model, subagents: list, *, system_prompt: str | None = None, **kwargs) -> Agent:
    """Build an orchestrator Agent whose tools are the given sub-agent tools."""
    return Agent(
        model,
        tools=subagents,
        system_prompt=system_prompt or _DEFAULT_SUPERVISOR_PROMPT,
        **kwargs,
    )
```

`framework/orchestration/__init__.py`:
```python
from framework.orchestration.supervisor import subagent, supervisor

__all__ = ["subagent", "supervisor"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestration.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Add a live orchestration test**

Append to `tests/test_orchestration.py`:
```python
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_supervisor_delegates_to_specialists():
    from framework import Agent
    from framework.orchestration import subagent, supervisor
    from framework.middleware import resilience

    MODEL = "groq:llama-3.3-70b-versatile"
    researcher = Agent(MODEL, system_prompt="You list 2-3 concise factual bullet points.")
    writer = Agent(MODEL, system_prompt="You turn bullet points into one clear sentence.")

    orch = supervisor(MODEL, [
        subagent(researcher, "researcher", "Gather facts on a topic as bullet points."),
        subagent(writer, "writer", "Turn facts into prose."),
    ], middleware=resilience())

    out = orch.run("Explain what an API is, briefly.")
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 6: Run the live test**

Run: `uv run pytest tests/test_orchestration.py -q`
Expected: PASS (2 passed) with a key present — the supervisor delegates and returns a synthesized answer.

---

### Task 2: Evaluation harness

**Files:**
- Create: `framework/eval/dataset.py`, `framework/eval/judge.py`, `framework/eval/runner.py`, `framework/eval/__init__.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `EVAL_SET: list[dict]` — cases with `type` `"trajectory"` (`expect_contains`, `expect_tools`) or `"judge"` (`rubric`).
  - `judge_answer(model, question, answer, rubric) -> {"score": int, "reason": str}`.
  - `score_case(agent, judge_model, case) -> dict` and `run_suite(agent, judge_model=None) -> list[dict]`.

- [ ] **Step 1: Write the failing unit test (no network — fake agent)**

`tests/test_eval.py`:
```python
from framework.eval.runner import score_case


class FakeAgent:
    def __init__(self, answer, events):
        self._a, self._e = answer, events

    def run_traced(self, message, thread_id=None):
        return {"answer": self._a, "summary": {}, "events": self._e}


def test_score_case_trajectory_pass():
    agent = FakeAgent("The result is 144.", [{"kind": "tool_call", "tool": "calculate"}])
    case = {"id": "m", "type": "trajectory", "input": "12*12?",
            "expect_contains": ["144"], "expect_tools": ["calculate"]}
    r = score_case(agent, None, case)
    assert r["passed"] is True


def test_score_case_trajectory_fails_on_missing_tool():
    agent = FakeAgent("The result is 144.", [])  # right answer, no tool call
    case = {"id": "m", "type": "trajectory", "input": "12*12?",
            "expect_contains": ["144"], "expect_tools": ["calculate"]}
    r = score_case(agent, None, case)
    assert r["passed"] is False  # trajectory check catches the lucky guess
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.eval.runner`.

- [ ] **Step 3: Write the dataset**

`framework/eval/dataset.py`:
```python
"""Eval cases with per-case judging: trajectory (outcome + tools) or LLM-judge."""

EVAL_SET = [
    {
        "id": "math-basic",
        "input": "What is 144 divided by 12?",
        "type": "trajectory",
        "expect_contains": ["12"],
        "expect_tools": ["calculate"],
    },
    {
        "id": "weather-lookup",
        "input": "What is the temperature in Lahore?",
        "type": "trajectory",
        "expect_contains": ["42"],
        "expect_tools": ["get_weather"],
    },
    {
        "id": "explain-agent",
        "input": "In two sentences, explain what an AI agent is to a beginner.",
        "type": "judge",
        "rubric": (
            "A good answer: accurate that an agent uses an LLM to take actions/use tools, "
            "beginner-friendly, roughly two sentences, no fluff."
        ),
    },
]
```

- [ ] **Step 4: Write the judge**

`framework/eval/judge.py`:
```python
"""LLM-as-judge — score an open-ended answer against a rubric, validated."""

from pydantic import BaseModel

from framework.structured import extract


class _Score(BaseModel):
    score: int
    reason: str


def judge_answer(model, question: str, answer: str, rubric: str) -> dict:
    prompt = f"Question:\n{question}\n\nRubric:\n{rubric}\n\nAnswer to score:\n{answer}"
    result = extract(
        model, prompt, _Score,
        instruction="Score the answer from 1-5 against the rubric. Be strict.",
    )
    return {"score": int(result.score), "reason": result.reason}
```

- [ ] **Step 5: Write the runner**

`framework/eval/runner.py`:
```python
"""Evaluation runner — scores a composed Agent on outcome AND trajectory.

Runs the real Agent.run_traced and reads the tool trajectory from its events, so
evaluation exercises the production code path. Open-ended cases go to an LLM judge.
"""

from framework.eval.dataset import EVAL_SET
from framework.eval.judge import judge_answer

JUDGE_PASS = 4


def _trajectory(events) -> list[str]:
    return [e["tool"] for e in events if e.get("kind") == "tool_call"]


def score_case(agent, judge_model, case) -> dict:
    result = agent.run_traced(case["input"])
    answer = result["answer"]
    traj = _trajectory(result["events"])

    if case["type"] == "judge":
        verdict = judge_answer(judge_model, case["input"], answer, case["rubric"])
        passed = verdict["score"] >= JUDGE_PASS
        detail = f"judge {verdict['score']}/5 — {verdict['reason']}"
    else:
        outcome_ok = all(s.lower() in answer.lower() for s in case.get("expect_contains", []))
        traj_ok = all(t in traj for t in case.get("expect_tools", []))
        passed = outcome_ok and traj_ok
        detail = f"outcome={'OK' if outcome_ok else 'FAIL'}, trajectory={'OK' if traj_ok else 'FAIL'}"

    return {"id": case["id"], "type": case["type"], "passed": passed,
            "detail": detail, "answer": answer, "trajectory": traj}


def run_suite(agent, judge_model=None) -> list[dict]:
    judge_model = judge_model or "groq:llama-3.3-70b-versatile"
    return [score_case(agent, judge_model, c) for c in EVAL_SET]
```

`framework/eval/__init__.py`:
```python
from framework.eval.dataset import EVAL_SET
from framework.eval.judge import judge_answer
from framework.eval.runner import score_case, run_suite

__all__ = ["EVAL_SET", "judge_answer", "score_case", "run_suite"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_eval.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Add a live suite test**

Append to `tests/test_eval.py`:
```python
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_run_suite_mostly_passes():
    from framework import Agent
    from framework.middleware import resilience
    from framework.tools import get_weather, calculate
    from framework.eval import run_suite

    agent = Agent("groq:llama-3.3-70b-versatile", tools=[get_weather, calculate],
                  system_prompt="Use tools to answer accurately, then reply concisely.",
                  middleware=resilience())
    results = run_suite(agent)
    passed = sum(r["passed"] for r in results)
    assert passed >= 2  # at least the two trajectory cases
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: all Phase 1–5 tests pass.

---

## Self-review

**Spec coverage (Phase 5):** orchestration (sub-agents-as-tools + supervisor) → Task 1; eval
harness scoring outcome + trajectory + judge → Task 2. Reuses `Agent.run_traced` and
`structured.extract`, so eval tests the production path and the judge is validated — consistent
with the spec's "reuse" intent.

**Placeholder scan:** all code shown; orchestration + eval unit tests use fakes (no network);
live tests are key-gated. No TBD/TODO.

**Type consistency:** `subagent(agent, name, description)`, `supervisor(model, subagents, ...)`,
`judge_answer(model, question, answer, rubric) -> {score, reason}`, `score_case(agent,
judge_model, case)`, `run_suite(agent, judge_model=None)` are consistent across tasks and match
the `run_traced` event shape (`{"kind": "tool_call", "tool": ...}`) from Phase 3.
