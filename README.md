<div align="center">

# 🤖 AI Agent

**A production-lite agentic AI system — from a single LLM call to a deployable, safe, observable agent.**

Provider-agnostic · tool-using · memory-backed · guardrailed · fully evaluated.

</div>

---

## What is this?

A complete, from-scratch agentic AI framework built one milestone at a time — no framework magic, every layer written and understood. It starts at a raw LLM call and ends at an agent you can serve over HTTP, with the safety, resilience, and observability you'd actually want in production.

One canonical agent (`ProductionAgent`) sits at the center. Around it is a library of composable capabilities you can wire in as needed: memory, RAG, MCP, orchestration, structured output, and an evaluation harness that tests the real agent.

```bash
uv run python api.py        # → open http://127.0.0.1:8001  (browser chat UI)
```

---

## ✨ Capabilities

Every layer below is present in this branch as working code.

| Layer | Capability | Where it lives |
|-------|-----------|----------------|
| 🧠 **Memory** | Multi-turn conversation history + sliding-window cap | `agent/memory/` |
| 📐 **Structure** | Strict JSON output + Pydantic-validated extraction | `agent/structured/` |
| 🔌 **Portability** | 5-provider abstraction (Groq · OpenAI · Anthropic · Google · OpenRouter) + streaming | `agent/llms/` |
| 🛠️ **Action** | Multi-tool ReAct loop (think → act → observe → repeat) | `agent/production/agent.py`, `agent/tools.py` |
| 📚 **Grounding** | RAG — chunk → embed → cosine search → augment | `agent/rag/` |
| 🔗 **Standards** | MCP client — discovers & calls tools over the protocol | `agent/mcp/`, `mcp_server/` |
| 🎭 **Coordination** | Orchestrator + specialist sub-agents (plan → delegate → synthesize) | `agent/orchestration/` |
| 🎯 **Rigor** | Eval harness scoring **outcome *and* trajectory** + LLM-as-judge | `agent/eval/` |
| 🛡️ **Safety** | Human-in-the-loop gates · per-request spend cap · prompt-injection defense | `agent/safety/` |
| 🚀 **Production** | Retry + provider fallback · structured trajectory logging · HTTP API | `agent/production/`, `api.py` |

---

## 🧭 Architecture

A single agent loop. Everything else is a library it can compose.

```
agent/
├── llms/                  Provider abstraction — one interface, five backends
│   ├── base.py              BaseLLM: send · stream · send_with_tools
│   ├── factory.py           get_llm("groq" | "openai" | "anthropic" | "google" | "openroute")
│   └── <provider>/          config.py (keys/model) + llm.py (the only SDK call site)
├── memory/                Conversation history + sliding-window (FIFO) cap
├── structured/            extract_json() + extract_model() (Pydantic-validated)
├── tools.py               Tool schemas + registry (read-only tools + destructive stubs)
├── safety/
│   └── guardrails.py        Per-request spend cap + destructive-tool approval gate
├── production/
│   ├── agent.py             ★ ProductionAgent — THE agent loop
│   ├── resilient_llm.py     Retry (exponential backoff) + provider fallback
│   └── observability.py     TrajectoryLogger — timed, structured event log
├── rag/                   chunker.py + vector_store.py (in-memory embeddings)
├── mcp/                   client.py — connect, discover, call tools over MCP
├── orchestration/         sub_agent.py + orchestrator.py
└── eval/                  dataset.py + judge.py + runner.py (tests the real agent)

api.py                     Deployable HTTP interface (browser UI + JSON API)
mcp_server/server.py       Example MCP server (company directory)
docs/                      Sample documents for the RAG pipeline
```

**One request, end to end:** `POST /chat` → spend budget reset → ReAct loop (each model call retried & fallback-guarded, each tool call authorized by the safety gate) → every step logged → JSON response with the answer *and* its full trajectory.

---

## 💡 Why it's built this way

- **Provider-agnostic by design.** Every model sits behind one `send`/`stream`/`send_with_tools` interface. Swapping Groq → Claude → Gemini is a one-line `.env` change — no app code touched.
- **One loop, not four.** A single canonical agent runs everything; the eval harness and the API drive that *same* code path, so tests exercise production behavior — not a parallel copy.
- **Safety in code, not vibes.** Destructive actions are blocked by a deterministic gate in the execution layer. Even if the model is fooled by a prompt injection, a denied approval means the function is *never called*. The guarantee doesn't depend on the model behaving.
- **Observable by default.** Every request returns a trajectory: model calls, tool calls, blocked calls, and total latency — so you can see exactly what the agent did and where the time and money went.
- **Resilient at the network edge.** Transient failures retry with backoff; a dead provider falls back to the next. Callers never see the churn.
- **Composable, not monolithic.** RAG, MCP, orchestration, memory, and structured output are independent modules. Use one, some, or all.

---

## 🚦 Production readiness — an honest assessment

This is **production-*lite*** — the architecture and control planes are real; the outermost integrations are demo-grade. Straight talk on what's ready and what isn't:

| Area | Status | Notes |
|------|:------:|-------|
| Provider abstraction & fallback | ✅ Ready | Battle-tested pattern, 5 providers |
| Safety gates & spend cap | ✅ Ready | Deterministic, per-request, verified |
| Observability / trajectory logging | ✅ Ready | Structured events; ship to a log aggregator as-is |
| Evaluation harness | ✅ Ready | Outcome + trajectory + LLM-judge; wire into CI |
| Multi-turn memory | 🟡 Local-only | Works, but the API uses **one shared** conversation — a real deploy needs per-session memory keyed by a session id |
| Tools | 🟡 Stubs | `get_weather`, `send_email`, `delete_record` return fake data — swap for real APIs |
| RAG vector store | 🟡 In-memory | Fine to start; move to a real vector DB (Chroma/Pinecone/pgvector) at scale |
| Concurrency | 🟡 Blocking | `agent.run()` is synchronous; run in a threadpool under load |
| Auth / rate-limiting | ❌ Not built | Add before exposing the API publicly |
| Prompt caching | ❌ Not built | Provider-side caching not yet wired for cost/latency |

**Bottom line:** the *skeleton* — safety, resilience, observability, evaluation, provider abstraction — is production-grade. To ship, replace the stub tools with real integrations, add per-session memory and auth, and move the vector store to a managed DB.

---

## 🏁 Quick start

**1. Install**
```bash
uv sync
```

**2. Configure** — create `.env` in the project root (git-ignored). Set the active provider + at least one key:
```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-key
GROQ_MODEL=llama-3.3-70b-versatile
# GOOGLE_API_KEY=...   # required only for RAG embeddings
# also supported: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTE_API_KEY
```

**3. Run the agent**
```bash
uv run python api.py
```
Open **http://127.0.0.1:8001** for the browser chat, or call the API directly:
```bash
curl -X POST localhost:8001/chat -H "content-type: application/json" \
  -d '{"message":"What is the weather in Lahore, and what is 15% of 240?"}'
```

**4. Run the eval suite**
```bash
uv run python -m agent.eval.runner
```

> **Port in use?** Set another: `set PORT=8002 && uv run python api.py` (Windows) — the server reads `HOST`/`PORT` from the environment.

---

## 🔍 Using the pieces directly

```python
from agent.llms.factory import get_llm
from agent.memory.conversation import ConversationMemory
from agent.safety.guardrails import Guardrails, console_approver
from agent.production.agent import ProductionAgent
from agent.production.resilient_llm import ResilientLLM

llm = get_llm("groq")
agent = ProductionAgent(
    ResilientLLM(providers=[("groq", llm)], max_retries=2),
    Guardrails(max_calls=8, approver=console_approver),   # asks a human before destructive tools
    memory=ConversationMemory(max_messages=20),           # multi-turn
)

result = agent.run("What is 15% of 240?")
print(result["answer"])
print(result["trace"])   # {model_calls, tool_calls, blocked_calls, total_ms}
```

**Structured output**
```python
from agent.structured.extract import extract_model
from pydantic import BaseModel

class Ticket(BaseModel):
    summary: str
    priority: str
    category: str

ticket = extract_model(llm, "Login returns 500s for all users since this morning.", Ticket)
# → Ticket(summary='...', priority='High', category='Authentication')
```

`rag/`, `mcp/`, and `orchestration/` compose the same way — see each module's docstring.

---

## 🛡️ Safety in action

Ask the browser agent to do something destructive:

> *"Email a summary to my boss at boss@corp.com."*

The agent will decline — `send_email` is a destructive tool, and the approval gate defaults to **deny**. That's the Milestone-10 guardrail, visible in the UI: the model can *request* the action, but the code decides whether it *executes*.

---

<div align="center">

*Built as a hands-on climb through 11 milestones — step by step*

</div>
