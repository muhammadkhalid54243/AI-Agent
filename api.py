"""Milestone 11 — the deployable interface.

A minimal HTTP API (starlette) exposing the ProductionAgent. This is what turns
a script into a service: something other than a Python REPL can now call the
agent over HTTP and get back a JSON answer plus its trajectory.

  POST /chat   {"message": "..."}   -> {request_id, status, answer, trace}
  GET  /health                      -> {"status": "ok"}

Run the server:
    python api.py
    # then: curl -X POST localhost:8000/chat -d '{"message":"weather in Lahore?"}'
"""

import os

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.llms.factory import get_llm
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
    return ProductionAgent(resilient, guards)


# Built once at startup and reused across requests (not per-request).
AGENT = build_agent()


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
    Route("/health", health, methods=["GET"]),
    Route("/chat", chat, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
