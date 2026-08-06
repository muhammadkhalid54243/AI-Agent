# LangGraph Framework — Phase 1 (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the modular core of a LangGraph-based agent framework — provider resolution with fallbacks, ported tools, and a composable `Agent` that runs a tool-using ReAct loop — as a new `framework/` package replacing plain-Python `agent/`.

**Architecture:** `Agent` is a thin composer over LangGraph's `create_agent` prebuilt. Provider models resolve via `init_chat_model("provider:model")` with `.with_fallbacks()` for resilience. Tools are LangChain `@tool` functions. Only `model` is required; everything else is opt-in (memory, safety, RAG, MCP arrive in later phases).

**Tech Stack:** Python ≥3.12, LangGraph 1.x, LangChain (core + provider integrations), pytest, uv.

## Global Constraints

- Python requires-python `>=3.12` (from `pyproject.toml`).
- Secrets come only from environment variables via `.env` / `python-dotenv`; never hardcoded, never committed.
- Pin LangGraph/LangChain deps to compatible releases (fewer-deprecations goal).
- Package name is `framework` (working name; rename is a later, cheap change).
- Every agentic loop keeps a hard cap (recursion/tool-call limit) — enforced fully in Phase 2; Phase 1 relies on `create_agent`'s default recursion limit.
- Supported providers and `init_chat_model` prefixes: `groq:`, `openai:`, `anthropic:`, `google_genai:`; OpenRouter via the `openai` provider + `base_url` override.

---

## Phase roadmap (context; only Phase 1 is planned in full here)

1. **Core (this plan)** — models + fallbacks, tools, `Agent` composer.
2. **Memory + Safety** — checkpointer/`thread_id`, guardrails middleware (`interrupt()` + tool-call cap).
3. **Observability + Structured output** — trajectory callback; `response_format`.
4. **Grounding + Standards** — RAG retriever tool; MCP loader.
5. **Coordination + Rigor** — supervisor/sub-agents; eval harness over the graph.
6. **Serve + UI + Polish** — FastAPI + SSE + browser UI; delete old `agent/`; docs.

Each subsequent phase gets its own plan after this one is green.

---

## File structure (Phase 1)

- Create `framework/__init__.py` — exports `Agent`.
- Create `framework/core/__init__.py` — empty package marker.
- Create `framework/core/models.py` — `parse_model_spec()`, `resolve_model()`.
- Create `framework/core/agent.py` — `Agent` class.
- Create `framework/tools/__init__.py` — re-exports built-in tools.
- Create `framework/tools/builtin.py` — `get_weather`, `calculate` as `@tool`.
- Create `tests/__init__.py` — empty.
- Create `tests/test_models.py`, `tests/test_tools.py`, `tests/test_agent.py`.
- Modify `pyproject.toml` — add LangGraph/LangChain deps + pytest dev group.

---

### Task 1: Dependencies and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `framework/__init__.py`, `framework/core/__init__.py`, `framework/tools/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: an importable `framework` package; `pytest` runnable.

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

Replace the `dependencies = [...]` array and add a dev group so the file's `[project]` block reads:

```toml
[project]
name = "ai-agent"
version = "0.1.0"
description = "A modular LangGraph-based agent framework."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.0,<2.0",
    "langchain>=1.0,<2.0",
    "langchain-groq>=0.3",
    "langchain-openai>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-google-genai>=2.0",
    "python-dotenv>=1.2.2",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 2: Create empty package markers**

Create these four files, each empty except `framework/__init__.py`:

`framework/__init__.py`:
```python
from framework.core.agent import Agent

__all__ = ["Agent"]
```

`framework/core/__init__.py`, `framework/tools/__init__.py`, `tests/__init__.py`: empty files.

- [ ] **Step 3: Sync dependencies**

Run: `uv sync`
Expected: resolves and installs langgraph, langchain, provider integrations, and pytest without error.

- [ ] **Step 4: Verify pytest runs**

Run: `uv run pytest -q`
Expected: `no tests ran` (exit 5) — pytest is installed and the package imports don't crash. (`framework/__init__.py` import of `Agent` will fail until Task 4; if so, temporarily comment the import, or implement Task 4 before running the full import — acceptable since Steps here only need pytest present.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml framework/ tests/__init__.py
git commit -m "chore: scaffold framework package and add LangGraph deps"
```

---

### Task 2: Model resolution with fallbacks

**Files:**
- Create: `framework/core/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `parse_model_spec(spec: str) -> tuple[str, dict]` — returns `(model_id, extra_kwargs)` where `model_id` is what `init_chat_model` accepts (e.g. `"groq:llama-3.3-70b-versatile"`) and `extra_kwargs` carries provider-specific extras (e.g. OpenRouter `base_url`).
  - `resolve_model(spec: str, fallbacks: list[str] | None = None)` — returns a LangChain chat model, with `.with_fallbacks([...])` applied when fallbacks are given.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from framework.core.models import parse_model_spec


def test_parse_standard_spec_passes_through():
    model_id, extra = parse_model_spec("groq:llama-3.3-70b-versatile")
    assert model_id == "groq:llama-3.3-70b-versatile"
    assert extra == {}


def test_parse_openrouter_rewrites_to_openai_with_base_url():
    model_id, extra = parse_model_spec("openrouter:meta-llama/llama-3.1-8b-instruct")
    assert model_id == "openai:meta-llama/llama-3.1-8b-instruct"
    assert extra["base_url"] == "https://openrouter.ai/api/v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.core.models` (or ImportError).

- [ ] **Step 3: Write minimal implementation**

`framework/core/models.py`:
```python
"""Provider resolution: 'provider:model' string -> a LangChain chat model.

Handles the five supported providers. OpenRouter is not a native init_chat_model
provider, so it is rewritten to the OpenAI provider with a base_url override.
Resilience (retry/fallback) is attached here via .with_fallbacks().
"""

from langchain.chat_models import init_chat_model

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_model_spec(spec: str) -> tuple[str, dict]:
    """Return (model_id_for_init_chat_model, extra_kwargs)."""
    provider, _, name = spec.partition(":")
    if provider == "openrouter":
        return f"openai:{name}", {"base_url": _OPENROUTER_BASE_URL}
    return spec, {}


def resolve_model(spec: str, fallbacks: list[str] | None = None):
    """Build a chat model from a spec, attaching fallbacks if provided."""
    model_id, extra = parse_model_spec(spec)
    model = init_chat_model(model_id, **extra)
    if fallbacks:
        fallback_models = [resolve_model(f) for f in fallbacks]
        model = model.with_fallbacks(fallback_models)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add framework/core/models.py tests/test_models.py
git commit -m "feat: model resolution with openrouter rewrite and fallbacks"
```

---

### Task 3: Port built-in tools to LangChain `@tool`

**Files:**
- Create: `framework/tools/builtin.py`
- Modify: `framework/tools/__init__.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `get_weather`, `calculate` — LangChain `BaseTool` objects (via `@tool`), each with `.name`, `.description`, and `.invoke(dict)`.

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import json
from framework.tools import get_weather, calculate


def test_get_weather_is_a_named_tool():
    assert get_weather.name == "get_weather"
    out = json.loads(get_weather.invoke({"city": "Lahore"}))
    assert out["city"] == "Lahore"
    assert out["temp_c"] == 42


def test_calculate_evaluates_and_rejects_bad_input():
    assert json.loads(calculate.invoke({"expression": "12 * 12"}))["result"] == 144
    assert "error" in json.loads(calculate.invoke({"expression": "__import__('os')"}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py -q`
Expected: FAIL with ImportError (`get_weather` not found).

- [ ] **Step 3: Write minimal implementation**

`framework/tools/builtin.py`:
```python
"""Built-in example tools, ported to LangChain @tool.

The tool *functions* carry over from the plain-Python version; the @tool
decorator turns them into LangChain BaseTool objects the agent can bind.
"""

import json

from langchain.tools import tool

_WEATHER = {
    "lahore": {"temp_c": 42, "condition": "sunny", "humidity": 30},
    "london": {"temp_c": 18, "condition": "cloudy", "humidity": 75},
    "tokyo": {"temp_c": 31, "condition": "humid", "humidity": 80},
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (temperature in Celsius, condition, humidity)."""
    data = _WEATHER.get(city.lower(), {"temp_c": 22, "condition": "unknown", "humidity": 50})
    return json.dumps({"city": city, **data})


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression (supports + - * / and parentheses)."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:  # noqa: BLE001 - surface any eval error as data
        return json.dumps({"error": str(e)})
```

`framework/tools/__init__.py`:
```python
from framework.tools.builtin import get_weather, calculate

__all__ = ["get_weather", "calculate"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add framework/tools/ tests/test_tools.py
git commit -m "feat: port built-in tools to LangChain @tool"
```

---

### Task 4: The `Agent` composer

**Files:**
- Create: `framework/core/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `resolve_model` (Task 2); `get_weather`, `calculate` (Task 3).
- Produces:
  - `Agent(model, *, tools=None, system_prompt=None, fallbacks=None)` — `model` is a spec string OR a pre-built chat model (the latter enables tests without network).
  - `Agent.add_tools(tools: list) -> None` — appends tools and rebuilds the graph.
  - `Agent.run(message: str, thread_id: str | None = None) -> str` — returns the final answer text.

- [ ] **Step 1: Write the failing test**

`tests/test_agent.py` (uses a fake tool-calling chat model so no network/keys are needed):
```python
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from framework import Agent
from framework.tools import calculate


class OneShotModel(GenericFakeChatModel):
    """Returns a plain answer with no tool calls — exercises the compose+invoke path."""


def test_agent_runs_and_returns_text_with_injected_model():
    fake = OneShotModel(messages=iter([AIMessage(content="The answer is 144.")]))
    agent = Agent(fake, tools=[calculate])
    out = agent.run("What is 12 times 12?")
    assert "144" in out


def test_add_tools_appends():
    fake = OneShotModel(messages=iter([AIMessage(content="ok")]))
    agent = Agent(fake, tools=[])
    agent.add_tools([calculate])
    assert calculate in agent.tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -q`
Expected: FAIL with ImportError (`Agent` cannot be imported / not defined).

- [ ] **Step 3: Write minimal implementation**

`framework/core/agent.py`:
```python
"""Agent — a thin, modular composer over LangGraph's create_agent prebuilt.

Only `model` is required. Tools, a system prompt, and fallbacks are optional;
memory, safety middleware, RAG, and MCP are added in later phases. `model` may
be a 'provider:model' spec string or an already-built chat model (handy for tests).
"""

from langchain.agents import create_agent

from framework.core.models import resolve_model


class Agent:
    def __init__(self, model, *, tools=None, system_prompt=None, fallbacks=None):
        self._model = model if not isinstance(model, str) else resolve_model(model, fallbacks)
        self.tools = list(tools) if tools else []
        self._system_prompt = system_prompt
        self._graph = self._build()

    def _build(self):
        kwargs = {"model": self._model, "tools": self.tools}
        if self._system_prompt:
            kwargs["prompt"] = self._system_prompt
        return create_agent(**kwargs)

    def add_tools(self, tools: list) -> None:
        self.tools.extend(tools)
        self._graph = self._build()

    def run(self, message: str, thread_id: str | None = None) -> str:
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        result = self._graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return result["messages"][-1].content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -q`
Expected: PASS (2 passed). If `create_agent`'s import path differs in the installed version, check `from langchain.agents import create_agent` vs `from langgraph.prebuilt import create_react_agent` and adjust the import + call in `_build` accordingly (see spec §9 open question).

- [ ] **Step 5: Commit**

```bash
git add framework/core/agent.py tests/test_agent.py
git commit -m "feat: Agent composer over create_agent"
```

---

### Task 5: Live integration smoke test + example

**Files:**
- Create: `tests/test_integration.py`
- Create: `examples/quickstart.py`

**Interfaces:**
- Consumes: `Agent` (Task 4), `get_weather`, `calculate` (Task 3).
- Produces: a runnable example and a network-gated integration test.

- [ ] **Step 1: Write the integration test (gated on an API key)**

`tests/test_integration.py`:
```python
import os
import pytest

from framework import Agent
from framework.tools import get_weather, calculate

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY for a live run"
)


def test_live_tool_use_answers_weather():
    agent = Agent(
        "groq:llama-3.3-70b-versatile",
        tools=[get_weather, calculate],
        system_prompt="Use tools to answer accurately. Then reply in one sentence.",
    )
    out = agent.run("What is the temperature in Lahore?")
    assert "42" in out
```

- [ ] **Step 2: Run the gated test with a key present**

Run: `uv run pytest tests/test_integration.py -q`
Expected: PASS if `GROQ_API_KEY` is set (agent calls `get_weather`, answers with 42); SKIPPED if not set.

- [ ] **Step 3: Write the example**

`examples/quickstart.py`:
```python
"""Minimal framework usage — compose an agent and ask it something.

    uv run python examples/quickstart.py
"""

from dotenv import load_dotenv

from framework import Agent
from framework.tools import get_weather, calculate

load_dotenv()


def main():
    agent = Agent(
        "groq:llama-3.3-70b-versatile",
        tools=[get_weather, calculate],
        system_prompt="Use tools to answer. Reply concisely.",
    )
    print(agent.run("What's the weather in Lahore, and what is 15% of 240?"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the example (with keys in .env)**

Run: `uv run python examples/quickstart.py`
Expected: prints a sentence containing the Lahore weather and the 36 result.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py examples/quickstart.py
git commit -m "test: live integration smoke test and quickstart example"
```

---

## Self-review

**Spec coverage (Phase 1 scope):** provider abstraction + fallbacks → Task 2; tools → Task 3;
ReAct loop via `create_agent` + modular composer → Task 4; runnable proof → Task 5. Memory, safety,
observability, structured output, RAG, MCP, orchestration, eval, and serve/UI are explicitly
deferred to Phases 2–6 (each with its own plan) — consistent with spec §7.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the one conditional
(`create_agent` import path) is a documented spec open question with an explicit fallback action,
not a placeholder.

**Type consistency:** `parse_model_spec`/`resolve_model` signatures match between Task 2 definition
and Task 4 consumption; `get_weather`/`calculate` names match across Tasks 3–5; `Agent(...)`,
`.add_tools`, `.run`, `.tools` are consistent across Tasks 4–5.

**Known deferrals carried to Phase 2:** the hard tool-call/spend cap (Phase 1 relies on
`create_agent`'s default recursion limit); `thread_id` is accepted by `run()` but only becomes
persistent memory once a checkpointer is added in Phase 2.
