"""Milestone 11 — the deployable interface.

A minimal HTTP API (starlette) exposing the ProductionAgent. This is what turns
a script into a service: something other than a Python REPL can now call the
agent over HTTP and get back a JSON answer plus its trajectory.

  GET  /            -> a minimal browser chat UI
  POST /chat   {"message": "..."}   -> {request_id, status, answer, trace}
  GET  /health                      -> {"status": "ok"}

Run the server:
    python api.py
    # then open http://127.0.0.1:8000 in a browser,
    # or: curl -X POST localhost:8000/chat -d '{"message":"weather in Lahore?"}'
"""

import os

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from agent.llms.factory import get_llm
from agent.memory.conversation import ConversationMemory
from agent.safety.guardrails import Guardrails, always_deny
from agent.production.agent import ProductionAgent
from agent.production.resilient_llm import ResilientLLM

load_dotenv()


def build_agent() -> ProductionAgent:
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    llm = get_llm(provider)
    resilient = ResilientLLM(providers=[(provider, llm)], max_retries=2)
    guards = Guardrails(max_calls=8, approver=always_deny)
    # NOTE: one shared memory for the whole server = a single global conversation.
    # Fine for local/single-user. A real multi-user deploy needs per-session memory
    # keyed by a session id, not one instance shared across all callers.
    memory = ConversationMemory(max_messages=20)
    return ProductionAgent(resilient, guards, memory=memory)


# Built once at startup and reused across requests (not per-request).
AGENT = build_agent()


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
    h1 { font-size: 1.3rem; }
    #log { border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 240px; margin-bottom: 12px; }
    .msg { margin: 8px 0; }
    .you { color: #1a5; }
    .bot { color: #222; white-space: pre-wrap; }
    .meta { color: #999; font-size: 0.8rem; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 8px 16px; border: 0; border-radius: 6px; background: #1a5; color: #fff; cursor: pointer; }
    button:disabled { background: #aaa; }
  </style>
</head>
<body>
  <h1>AI Agent</h1>
  <div id="log"></div>
  <form id="f">
    <input id="m" placeholder="Ask something, e.g. What's the weather in Lahore?" autofocus>
    <button id="b">Send</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('f');
    const input = document.getElementById('m');
    const btn = document.getElementById('b');
    function add(cls, text) {
      const d = document.createElement('div');
      d.className = 'msg ' + cls;
      d.textContent = text;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
      return d;
    }
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      add('you', 'You: ' + message);
      input.value = '';
      btn.disabled = true;
      const pending = add('bot', 'Nova: ...');
      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await r.json();
        pending.textContent = 'Nova: ' + data.answer;
        const t = data.trace || {};
        add('meta', `(${t.model_calls} model calls, ${t.tool_calls} tool calls, ${t.blocked_calls} blocked, ${t.total_ms} ms)`);
      } catch (err) {
        pending.textContent = 'Nova: [error] ' + err;
      } finally {
        btn.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>"""


async def index(request):
    return HTMLResponse(INDEX_HTML)


async def health(request):
    return JSONResponse({"status": "ok"})


async def chat(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    message = (body or {}).get("message")
    if not message:
        return JSONResponse({"error": "missing 'message' field"}, status_code=400)

    # run() is synchronous/blocking; in a real deploy run it in a threadpool.
    result = AGENT.run(message)
    return JSONResponse(result)


app = Starlette(routes=[
    Route("/", index, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/chat", chat, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
