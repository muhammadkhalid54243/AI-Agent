<div align="center">

# 🤖 Agent Framework

**A modular, LangGraph-based agent framework — compose exactly the capabilities you need.**

Provider-agnostic · memory · tools · RAG · MCP · orchestration · safety · eval · streaming UI

</div>

---

## What is this?

A small framework built on **LangGraph 1.x**. One composable `Agent` sits at the center; every
capability is an **opt-in module** you add as needed — nothing is mandatory except a model. It
rides LangGraph's stable prebuilts (fewer deprecations) while keeping each capability behind a
thin seam of its own.

```python
from framework import Agent
from framework.memory import checkpoint
from framework.middleware import guardrails, resilience
from framework.tools import get_weather, calculate
from framework.rag import Retriever
from framework.mcp import load_mcp_tools

agent = Agent(
    "groq:llama-3.3-70b-versatile",
    fallbacks=["openai:gpt-4o-mini"],                       # provider fallback
    tools=[get_weather, calculate],
    memory=checkpoint("memory"),                            # per-session via thread_id
    middleware=guardrails(require_approval=["send_email"],  # human-in-the-loop gate
                          model_call_limit=8)               # spend cap
              + resilience(),                               # retry transient errors
    system_prompt="You are Nova. Be concise.",
)

agent.add_tools([Retriever("docs/").as_tool()])             # RAG, opt-in
agent.add_tools(load_mcp_tools({"dir": {...}}))             # MCP, opt-in

print(agent.run("What's the weather in Lahore?", thread_id="user-1"))
for token in agent.stream("Tell me a joke", thread_id="user-1"):
    print(token, end="")
```

---

## ✨ Capabilities

| Capability | Module | Backed by |
|-----------|--------|-----------|
| 🔌 **Providers + fallback** | `framework.core` | `init_chat_model` + `.with_fallbacks()` |
| 🧠 **Memory (per-session)** | `framework.memory` | checkpointer + `thread_id` |
| 🛠️ **Tools + ReAct loop** | `framework.tools`, `core` | `create_agent` |
| 🛡️ **Safety** (approval gate, spend cap) | `framework.middleware` | `HumanInTheLoop` + `ModelCallLimit` middleware |
| ♻️ **Resilience** (retry/backoff) | `framework.middleware` | `ModelRetryMiddleware` |
| 📐 **Structured output** | `framework.structured` | `with_structured_output` + Pydantic |
| 🔭 **Observability** | `framework.observability` | callback tracer → `run_traced()` |
| 📚 **RAG** | `framework.rag` | vector store + retriever tool |
| 🔗 **MCP** | `framework.mcp` | `langchain-mcp-adapters` |
| 🎭 **Orchestration** | `framework.orchestration` | sub-agents-as-tools + supervisor |
| 🎯 **Evaluation** | `framework.eval` | outcome + trajectory + LLM-judge |
| 🚀 **Serving + UI** | `framework.serve` | starlette + SSE streaming + browser chat |

---

## 🧭 Architecture

```
framework/
├── core/            Agent composer + provider/model resolution (fallbacks)
├── memory/          checkpoint() — per-session memory keyed by thread_id
├── tools/           built-in + destructive @tool functions
├── middleware/      guardrails() (safety) + resilience() (retry)
├── structured/      extract() — validated Pydantic output
├── observability/   TrajectoryTracer — timed model/tool events
├── rag/             Retriever(...).as_tool()
├── mcp/             load_mcp_tools(config)
├── orchestration/   subagent() + supervisor()
├── eval/            dataset + judge + run_suite()
└── serve/           app (HTTP + SSE) + browser UI
serve.py             run the chat server
examples/demo.py     see everything working
mcp_server/          example MCP server (company directory)
tests/               TDD suite (unit + key-gated live integration)
```

`Agent` calls LangGraph's `create_agent` and wires in whatever modules you pass. Because every
module produces standard LangChain objects (tools, middleware, checkpointers), composition is
uniform and swapping providers is a one-line change.

---

## 🏁 Quick start

```bash
uv sync
```

Create `.env` (git-ignored) with at least one provider key:

```dotenv
GROQ_API_KEY=your-key
# FRAMEWORK_MODEL=groq:llama-3.3-70b-versatile   # optional override
# GOOGLE_API_KEY=...   # only for RAG embeddings
# also usable: OPENAI_API_KEY, ANTHROPIC_API_KEY
```

**Run the chat server (streaming browser UI):**
```bash
uv run python serve.py      # → open http://127.0.0.1:8000
```

**See every capability in one script:**
```bash
uv run python examples/demo.py
```

**Run the tests (unit always; live integration when a key is set):**
```bash
uv run pytest -q
```

**Evaluate an agent:**
```python
from framework import Agent
from framework.middleware import resilience
from framework.tools import get_weather, calculate
from framework.eval import run_suite

agent = Agent("groq:llama-3.3-70b-versatile", tools=[get_weather, calculate],
              middleware=resilience())
for r in run_suite(agent):
    print(r["id"], "PASS" if r["passed"] else "FAIL", "-", r["detail"])
```

---

## 🛡️ Safety in action

Give the agent a destructive tool behind an approval gate:

```python
from framework.middleware import guardrails, ApprovalRequired
from framework.tools import delete_record
from framework.memory import checkpoint

agent = Agent("groq:llama-3.3-70b-versatile", tools=[delete_record],
              memory=checkpoint("memory"),
              middleware=guardrails(require_approval=["delete_record"]))

try:
    agent.run("Delete record 42.", thread_id="s1")
except ApprovalRequired as a:
    print(a.requests)                 # [{'name': 'delete_record', 'args': {'record_id': '42'}}]
agent.resume("s1", approve=False)     # → the delete never executes
```

The guarantee lives in the graph's control flow (an `interrupt()` before the side effect), not in
the prompt — a denied approval means the tool body never runs.

---

## ⚠️ Honest notes

- **`serve/` uses one shared memory** keyed by a client thread_id — fine for local/single-user; a
  multi-user deploy needs per-authenticated-session scoping and auth (not built).
- **Example tools are stubs** (`get_weather`, `send_email`, `delete_record`) — swap for real APIs.
- **Persistent checkpointers** (sqlite/postgres) are a planned extension; `checkpoint("memory")`
  is process-local.
- LangGraph is pinned (`>=1.0,<2.0`) for stability; upgrades are opt-in.

---

<div align="center">

*Built by porting a from-scratch agent onto LangGraph — design & phase plans in
[`docs/superpowers/`](docs/superpowers/).*

</div>
