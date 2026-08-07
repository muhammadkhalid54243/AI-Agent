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
    if body is None or not body.get("message"):
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
