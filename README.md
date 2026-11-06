<div align="center">

# 🤖 Agent Framework

### A modular, production-minded agent framework built on LangGraph

*Compose exactly the capabilities you need — memory, tools, RAG, MCP, orchestration, safety, evaluation, and a streaming chat UI — each an opt-in module, nothing mandatory but a model.*

<br>

![Python](https://img.shields.io/badge/python-3.12+-blue)
![LangGraph](https://img.shields.io/badge/built%20on-LangGraph%201.x-1a5)
![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

## Why this exists

Most agent code is either a thin wrapper around one provider or a monolith you can't take apart.
This framework is neither. **Every capability was first implemented from scratch in plain Python
— then ported onto LangGraph's stable primitives.** The result: abstractions that are *understood*,
not just imported, wrapped in a clean, composable API.

If you want to see how a real agent stack fits together — provider abstraction, a ReAct loop,
per-session memory, human-in-the-loop safety, retries, observability, RAG, MCP, multi-agent
orchestration, and evaluation — this is a compact, readable, tested example of all of it.

---

## Quick look

```python
from framework import Agent
from framework.memory import checkpoint
from framework.middleware import guardrails, resilience
from framework.tools import get_weather, calculate

agent = Agent(
    "groq:openai/gpt-oss-120b",
    fallbacks=["openai:gpt-4o-mini"],                       # automatic provider fallback
    tools=[get_weather, calculate],
    memory=checkpoint("memory"),                            # per-session memory (thread_id)
    middleware=guardrails(model_call_limit=8) + resilience(),  # spend cap + retry
    system_prompt="You are Nova. Be concise.",
)

print(agent.run("What's the weather in Lahore?", thread_id="user-1"))
for token in agent.stream("Tell me a joke", thread_id="user-1"):
    print(token, end="")     # streams token by token
```

Add more capabilities only when you need them:

```python
from framework.rag import Retriever
from framework.mcp import load_mcp_tools

agent.add_tools([Retriever("docs/").as_tool()])            # RAG grounding
agent.add_tools(load_mcp_tools({"dir": {...}}))            # tools discovered over MCP
```

---

## What's inside

| Capability | Module | How it's done |
|-----------|--------|---------------|
| 🔌 **Multi-provider + fallback** | `framework.core` | `init_chat_model` + `.with_fallbacks()` — swap Groq/OpenAI/Anthropic/Google in one line |
| 🧠 **Per-session memory** | `framework.memory` | LangGraph checkpointer keyed by `thread_id` |
| 🛠️ **Tools + ReAct loop** | `framework.tools`, `core` | `create_agent` — think → act → observe |
| 🛡️ **Human-in-the-loop safety** | `framework.middleware` | `interrupt()` before destructive tools — a denied approval means it *never runs* |
| 💸 **Spend cap** | `framework.middleware` | model-call limit per run |
| ♻️ **Resilience** | `framework.middleware` | retry with exponential backoff |
| 📐 **Structured output** | `framework.structured` | native `with_structured_output` + Pydantic validation |
| 🔭 **Observability** | `framework.observability` | callback tracer → `run_traced()` returns a full trajectory; **OpenTelemetry response tracing** (`tracing.py`) spans the served RAG pipeline — see below |
| 📚 **RAG** | `framework.rag` | vector store → retriever exposed as a tool |
| 🔗 **MCP** | `framework.mcp` | discover & call tools over the Model Context Protocol |
| 🎭 **Orchestration** | `framework.orchestration` | sub-agents-as-tools + a supervisor |
| 🎯 **Evaluation** | `framework.eval` | scores **outcome _and_ trajectory** + LLM-as-judge |
| 🚀 **Serving + UI** | `framework.serve` | HTTP + SSE streaming + a browser chat page |

---

## Getting started

**1. Install** (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

**2. Configure** — copy the example env and add at least one provider key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY (or another provider)
```

**3. Run the streaming chat server:**

```bash
uv run python serve.py      # → open http://127.0.0.1:8000
```

**4. Or see every capability in one script:**

```bash
uv run python examples/demo.py
```

**5. Run the tests:**

```bash
uv run pytest -q            # unit tests always; live tests run when a key is set
```

**6. (Optional) Response tracing with OpenTelemetry** — see each request as a trace
(`rag_answer → condense → retrieve → generate → parse → verify`) with per-stage latency and
attributes (n_hits, confidence, abstained, tokens). Off by default; self-hosted, nothing leaves
your machine. Quickest view is a local Jaeger:

```bash
docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
TRACE_BACKEND=otel OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run python serve.py
# ask a question, then open the Jaeger UI at http://localhost:16686
```

With no `OTEL_EXPORTER_OTLP_ENDPOINT`, spans print to the console instead. Sensitive text payloads
(question/answer/quote) are redacted unless `TRACE_PAYLOADS=1`. See `.env.example` for all knobs.

**7. (Optional) Metrics dashboard** — Jaeger shows one request; the dashboard shows the aggregate
over time: LLM calls by stage, tokens, per-stage latency (p50/p95), abstain rate, retrieval
confidence. Also off by default and fully self-hosted. One compose file runs Jaeger + Prometheus +
Grafana together:

```bash
# one-time: if you ever started Jaeger by hand, drop it so compose can own ports 4318/16686
docker rm -f jaeger
docker compose -f docker-compose.observability.yml up -d
TRACE_BACKEND=otel OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
    METRICS_BACKEND=prometheus uv run python serve.py
# ask a few questions, then open Grafana at http://localhost:3000 -> dashboard "HRCP RAG"
```

Lifecycle (metrics + Grafana live in `./observability/data/`, gitignored, so the **dashboard
survives restarts**; Jaeger traces are in-memory and reset on restart):

```bash
docker compose -f docker-compose.observability.yml stop    # pause (keeps containers + data)
docker compose -f docker-compose.observability.yml start   # resume
docker compose -f docker-compose.observability.yml down    # remove containers; DATA KEPT
# wipe data too: run `down`, then delete ./observability/data/{prometheus,grafana}/*
```

`METRICS_BACKEND=prometheus` exposes metrics on `METRICS_PORT` (default 9464) for Prometheus to
scrape (`curl localhost:9464/metrics` to see them raw). Metrics are counts/latency only — no text
payloads — so they carry no corpus content. Metrics and tracing are independent: enable either or
both. Prometheus/Jaeger only hold data for the windows when the server was running with these flags
— a restart shows the saved history, not the gaps when it was off.

---

## A few things worth seeing

**Human-in-the-loop safety** — the guarantee lives in the graph's control flow, not the prompt:

```python
from framework.middleware import guardrails, ApprovalRequired
from framework.tools import delete_record
from framework.memory import checkpoint

agent = Agent("groq:openai/gpt-oss-120b", tools=[delete_record],
              memory=checkpoint("memory"),
              middleware=guardrails(require_approval=["delete_record"]))
try:
    agent.run("Delete record 42.", thread_id="s1")
except ApprovalRequired as a:
    print(a.requests)              # [{'name': 'delete_record', 'args': {'record_id': '42'}}]
agent.resume("s1", approve=False)  # the delete never executes
```

**Evaluation that catches lucky guesses** — a right answer via the wrong path still fails:

```python
from framework.eval import run_suite
for r in run_suite(agent):
    print(r["id"], "PASS" if r["passed"] else "FAIL", "-", r["detail"])
```

**Structured, validated output:**

```python
from framework.structured import extract
from pydantic import BaseModel

class Ticket(BaseModel):
    summary: str
    priority: str

extract("groq:openai/gpt-oss-120b", "Login 500s for all users. Urgent.", Ticket)
# → Ticket(summary='...', priority='high')
```

---

## Architecture

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
```

`Agent` builds on LangGraph's `create_agent` and wires in whatever modules you pass. Because every
module produces standard LangChain objects (tools, middleware, checkpointers), composition is
uniform — and the whole thing is pinned to LangGraph `>=1.0,<2.0` for stability.

The full design spec and phase-by-phase implementation plans live in
[`docs/superpowers/`](docs/superpowers/).

---

## Honest limitations

This is a focused, readable framework — not a hosted platform. Known edges:

- The bundled **example tools are stubs** (`get_weather`, `send_email`, `delete_record`) — swap for real APIs.
- **`serve/` uses one shared conversation** keyed by a client thread id — great for local/single-user; a multi-user deployment needs per-authenticated-session scoping and auth.
- **Memory is in-process**; persistent checkpointers (SQLite/Postgres) are a planned drop-in.

---

## License

MIT — see [`LICENSE`](LICENSE).
