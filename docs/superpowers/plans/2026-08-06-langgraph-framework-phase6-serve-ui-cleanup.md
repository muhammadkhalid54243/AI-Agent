# LangGraph Framework — Phase 6 (Serve + UI + Cleanup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship a serving layer (HTTP + SSE streaming + browser chat UI with an approval flow), then delete the old plain-Python `agent/` package and write the framework README.

**Architecture:** `Agent.stream()` yields answer tokens from the graph. `serve/app.py` (starlette) exposes `GET /` (UI), `GET /health`, `POST /chat` (full answer + trace, or approval-required), `GET /chat/stream` (SSE tokens), `POST /approve` (resume). The UI streams responses and renders the approve/deny prompt. Old `agent/` + `api.py` are removed.

**Tech Stack:** starlette + uvicorn (already present), SSE via `StreamingResponse`, the framework `Agent`, pytest + starlette `TestClient`.

## Global Constraints

- Python `>=3.12`; secrets from env via `.env`; server reads `HOST`/`PORT`.
- Verified API facts (probed live): `agent._graph.stream({"messages":[...]}, stream_mode="messages")` yields `(chunk, meta)`; `chunk.content` holds token text.
- Build on Phases 1–5: `Agent` (`run_traced`, `resume`), `checkpoint`, `guardrails`, `resilience`, `ApprovalRequired`, `framework.tools`.

---

## File structure (Phase 6)

- Modify `framework/core/agent.py` — add `stream()`.
- Create `framework/serve/__init__.py` — re-exports `app`, `build_default_agent`.
- Create `framework/serve/app.py` — starlette app + routes.
- Create `framework/serve/ui.py` — `INDEX_HTML`.
- Create `serve.py` (repo root) — entrypoint `uv run python serve.py`.
- Test: `tests/test_serve.py`.
- Delete: `agent/` (whole package), `api.py`.
- Modify: `pyproject.toml` (drop plain-Python-only deps), `README.md`.

---

### Task 1: Agent.stream()

**Files:**
- Modify: `framework/core/agent.py`
- Test: `tests/test_serve.py` (stream part)

**Interfaces:**
- Produces: `Agent.stream(message, thread_id=None)` — a generator yielding answer token strings.

- [ ] **Step 1: Write the failing test (live)**

`tests/test_serve.py`:
```python
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_agent_stream_yields_tokens():
    from framework import Agent
    agent = Agent("groq:llama-3.3-70b-versatile", system_prompt="Be concise.")
    tokens = list(agent.stream("Say hello in four words."))
    assert len(tokens) >= 1
    assert "".join(tokens).strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_serve.py::test_agent_stream_yields_tokens -q`
Expected: FAIL — `Agent` has no `stream`.

- [ ] **Step 3: Add stream() to Agent**

In `framework/core/agent.py`, add after `run_traced`:
```python
    def stream(self, message: str, thread_id: str | None = None):
        """Yield answer tokens as they are generated (typewriter streaming)."""
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        for chunk, _meta in self._graph.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages",
        ):
            text = getattr(chunk, "content", "")
            if text:
                yield text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_serve.py::test_agent_stream_yields_tokens -q`
Expected: PASS with a key present.

---

### Task 2: serve/app.py + UI

**Files:**
- Create: `framework/serve/app.py`, `framework/serve/ui.py`, `framework/serve/__init__.py`, `serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Produces:
  - `build_default_agent() -> Agent` — read-only tools + memory + resilience.
  - `app` — a starlette `Starlette` with routes `/`, `/health`, `/chat`, `/chat/stream`, `/approve`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serve.py`:
```python
def test_health_and_ui_routes():
    from starlette.testclient import TestClient
    from framework.serve.app import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.get("/")
    assert r.status_code == 200 and "<form" in r.text


def test_chat_requires_message():
    from starlette.testclient import TestClient
    from framework.serve.app import app
    c = TestClient(app)
    assert c.post("/chat", json={}).status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_serve.py -q`
Expected: FAIL — `framework.serve.app` does not exist.

- [ ] **Step 3: Write the UI**

`framework/serve/ui.py`:
```python
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Framework Agent</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
 #log{border:1px solid #ddd;border-radius:8px;padding:12px;min-height:240px;margin-bottom:12px}
 .msg{margin:8px 0}.you{color:#1a5}.bot{color:#222;white-space:pre-wrap}.meta{color:#999;font-size:.8rem}
 form{display:flex;gap:8px}input{flex:1;padding:8px;border:1px solid #ccc;border-radius:6px}
 button{padding:8px 16px;border:0;border-radius:6px;background:#1a5;color:#fff;cursor:pointer}
</style></head><body>
<h1>Framework Agent</h1><div id="log"></div>
<form id="f"><input id="m" placeholder="Ask something..." autofocus><button>Send</button></form>
<script>
const log=document.getElementById('log'),form=document.getElementById('f'),input=document.getElementById('m');
const tid='web-'+Math.random().toString(36).slice(2,8);
function add(c,t){const d=document.createElement('div');d.className='msg '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
form.onsubmit=async e=>{
 e.preventDefault();const msg=input.value.trim();if(!msg)return;
 add('you','You: '+msg);input.value='';
 const bot=add('bot','Nova: ');
 const res=await fetch('/chat/stream?thread_id='+tid,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:msg})});
 const reader=res.body.getReader(),dec=new TextDecoder();
 while(true){const {value,done}=await reader.read();if(done)break;
  dec.decode(value).split('\\n\\n').forEach(line=>{if(line.startsWith('data: '))bot.textContent+=line.slice(6);});}
};
</script></body></html>"""
```

- [ ] **Step 4: Write the app**

`framework/serve/app.py`:
```python
"""Serving layer — HTTP + SSE streaming + a browser chat UI for a framework Agent.

Routes:
  GET  /              browser chat UI (streams responses)
  GET  /health        {"status": "ok"}
  POST /chat          {message, thread_id?} -> {status:"ok", answer, trace}
                      or {status:"approval_required", requests, thread_id}
  POST /chat/stream   {message} ?thread_id=  -> SSE token stream
  POST /approve       {thread_id, approve}   -> {answer} or approval_required

NOTE: one process-global Agent with shared memory keyed by thread_id from the
client. Fine for local/single-user; a real multi-user deploy scopes agents/keys
per authenticated session. run() is blocking; wrap in a threadpool under load.
"""

import os

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from framework import Agent
from framework.memory import checkpoint
from framework.middleware import resilience, ApprovalRequired
from framework.tools import get_weather, calculate
from framework.serve.ui import INDEX_HTML

load_dotenv()


def build_default_agent() -> Agent:
    model = os.environ.get("FRAMEWORK_MODEL", "groq:llama-3.3-70b-versatile")
    return Agent(model, tools=[get_weather, calculate],
                 system_prompt="You are Nova. Use tools when helpful. Be concise.",
                 memory=checkpoint("memory"), middleware=resilience())


AGENT = build_default_agent()


async def index(request):
    return HTMLResponse(INDEX_HTML)


async def health(request):
    return JSONResponse({"status": "ok"})


async def chat(request):
    body = await _json(request)
    if body is None:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    message = body.get("message")
    if not message:
        return JSONResponse({"error": "missing 'message'"}, status_code=400)
    thread_id = body.get("thread_id")
    try:
        result = AGENT.run_traced(message, thread_id=thread_id)
        return JSONResponse({"status": "ok", **result})
    except ApprovalRequired as a:
        return JSONResponse({"status": "approval_required",
                             "requests": a.requests, "thread_id": a.thread_id})


async def chat_stream(request):
    body = await _json(request)
    if body is None or not (body.get("message")):
        return JSONResponse({"error": "missing 'message'"}, status_code=400)
    message = body["message"]
    thread_id = request.query_params.get("thread_id")

    def sse():
        try:
            for token in AGENT.stream(message, thread_id=thread_id):
                yield f"data: {token}\n\n"
        except ApprovalRequired as a:
            yield f"data: [approval required for {a.requests}]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


async def approve(request):
    body = await _json(request)
    if body is None or "thread_id" not in body:
        return JSONResponse({"error": "missing 'thread_id'"}, status_code=400)
    try:
        answer = AGENT.resume(body["thread_id"], approve=bool(body.get("approve")))
        return JSONResponse({"status": "ok", "answer": answer})
    except ApprovalRequired as a:
        return JSONResponse({"status": "approval_required",
                             "requests": a.requests, "thread_id": a.thread_id})


async def _json(request):
    try:
        return await request.json()
    except Exception:
        return None


app = Starlette(routes=[
    Route("/", index, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/chat", chat, methods=["POST"]),
    Route("/chat/stream", chat_stream, methods=["POST"]),
    Route("/approve", approve, methods=["POST"]),
])
```

`framework/serve/__init__.py`:
```python
from framework.serve.app import app, build_default_agent

__all__ = ["app", "build_default_agent"]
```

`serve.py` (repo root):
```python
"""Run the framework's chat server.

    uv run python serve.py     # then open http://127.0.0.1:8000
"""

import os

if __name__ == "__main__":
    import uvicorn
    from framework.serve.app import app
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_serve.py -q`
Expected: PASS (health/UI + 400 validation; plus the live stream test if a key is set).

---

### Task 3: Delete the old plain-Python `agent/` package

**Files:**
- Delete: `agent/` (recursively), `api.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing (verify no framework code imports `agent`).

- [ ] **Step 1: Verify nothing in framework/tests imports the old package**

Run: `grep -rEn "from agent|import agent\b|agent\." framework tests serve.py 2>/dev/null || echo CLEAN`
Expected: `CLEAN` (only `framework.*` imports exist).

- [ ] **Step 2: Delete the old package and entrypoint**

Run:
```bash
rm -rf agent api.py
```

- [ ] **Step 3: Drop plain-Python-only deps from pyproject.toml**

In `pyproject.toml`, remove the `mcp[cli]` line and its comment (the plain-Python agent used it; MCP now comes via `langchain-mcp-adapters`). Keep everything else. The `mcp_server/server.py` example still needs `mcp` — it is pulled transitively by `langchain-mcp-adapters`, so the example keeps working; verify in Step 4.

- [ ] **Step 4: Sync and run the full suite**

Run: `uv sync && uv run pytest -q`
Expected: all framework tests pass; `tests/test_mcp.py` (which spawns `mcp_server/server.py`) still passes, confirming `mcp` remains importable transitively.

- [ ] **Step 5: Commit checkpoint (optional — leave to user)**

Skip if the user is driving git.

---

### Task 4: Framework README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README for the framework**

Replace `README.md` with framework-focused content: what it is (a modular LangGraph agent framework), the composition example, the capability→module table, quick start (`uv sync`; `.env`; `uv run python serve.py` → `http://127.0.0.1:8000`; `uv run python examples/demo.py`; `uv run pytest`), and an honest note that `serve` uses one shared memory (single-user). Include the `Agent(...)` example with tools, memory, guardrails, resilience.

- [ ] **Step 2: Verify the demo and server still run**

Run: `uv run python examples/demo.py` → prints the four sections.
Run (manual): `uv run python serve.py` then open `http://127.0.0.1:8000` and send a message → streamed reply.

---

## Self-review

**Spec coverage (Phase 6):** serving + UI + SSE streaming + approval endpoints → Tasks 1–2;
delete old `agent/` (this branch becomes the framework) → Task 3; framework docs → Task 4.

**Placeholder scan:** all code shown; server tests use `TestClient` (health/UI/validation are
deterministic, no key); the stream test is key-gated. README content is described concretely in
Task 4 Step 1 (rewrite, not a stub). No TBD/TODO.

**Type consistency:** `Agent.stream(message, thread_id)` generator, `build_default_agent() ->
Agent`, `app` routes, and the `ApprovalRequired(requests, thread_id)` shape are consistent across
tasks and match Phases 2–3 (`run_traced` dict, `resume`).
