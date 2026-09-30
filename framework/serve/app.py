"""Serving layer — HTTP + SSE streaming + a browser chat UI for a framework Agent.

Routes:
  GET  /              browser chat UI (streams responses)
  GET  /health        {"status": "ok"}
  POST /chat          {message, thread_id?} -> {status:"ok", answer, trace}
                      or {status:"approval_required", requests, thread_id}
  POST /chat/stream   {message} ?thread_id=  -> SSE token stream
  POST /approve       {thread_id, approve}   -> {answer} or approval_required

NOTE: one process-global Agent (built LAZILY on first chat via get_agent) with shared
memory keyed by thread_id from the client. Fine for local/single-user; a real
multi-user deploy scopes agents/keys per authenticated session. run() is blocking;
wrap in a threadpool under load.

RAG: get_state builds a CorpusRetriever over one or more ingested reports and answers
via retrieve-then-answer. Lazy so import + /health stay cheap and the heavy index/model
load is deferred to first chat. Config via env: RAG_DOCS (a directory of PDFs or a
comma-separated list; default docs/SHR), RAG_DOC (single-path override for back-compat),
RAG_RERANK=1 to enable the cross-encoder reranker (better ranking, slower/heavier).
"""

import os

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from framework import Agent
from framework.memory import checkpoint
from framework.middleware import resilience, ApprovalRequired
from framework.middleware.guardrails import guardrails
from framework.tools import get_weather, calculate
from framework.serve.ui import INDEX_HTML

# Hard cap for the base-agent fallback path (non-RAG); the RAG path is a single
# model call, so it needs no loop cap. Non-negotiable per CLAUDE.md: loops are capped.
MODEL_CALL_LIMIT = int(os.environ.get("RAG_MODEL_CALL_LIMIT", "6"))

load_dotenv()


FRAMEWORK_MODEL = os.environ.get("FRAMEWORK_MODEL", "groq:openai/gpt-oss-120b")
# RAG_DOCS: a directory (ingest every *.pdf in it) or a comma-separated list of paths.
# RAG_DOC (single path) still wins if set, for back-compat with the old single-doc serve.
RAG_DOCS = os.environ.get("RAG_DOCS", "docs/SHR")
RAG_K = int(os.environ.get("RAG_K", "8"))
NOT_COVERED = ("The reports don't appear to cover that. Try a more specific fact, figure, "
               "or section.")
RAG_SYS = (
    "You answer questions about HRCP human-rights reports using ONLY the passages below. "
    "Each passage begins with its source line: Report Year — section, pN.\n"
    "Rules:\n"
    "- Put each distinct fact on its OWN line.\n"
    "- End every line with its source in brackets, copied from that passage's source line: "
    "[Report Year, pN] — e.g. [SHR 2024, p18]. Never cite a bare passage number like [3].\n"
    "- These reports are yearly, so the same figure differs across years. When more than "
    "one report or year is relevant, give EACH on its own line with its own citation — "
    "never merge them into one number or silently pick just one year.\n"
    "- Use only what the passages state; add nothing. If they don't cover the question, "
    "say the reports don't appear to cover it.\n"
    "Be concise.")

_state = None


def _resolve_doc_paths() -> list:
    """Which documents to serve. A single RAG_DOC overrides; otherwise RAG_DOCS is a
    directory of PDFs or a comma-separated list of paths."""
    single = os.environ.get("RAG_DOC")
    if single:
        return [single]
    if os.path.isdir(RAG_DOCS):
        return sorted(os.path.join(RAG_DOCS, f) for f in os.listdir(RAG_DOCS)
                      if f.lower().endswith(".pdf"))
    return [p.strip() for p in RAG_DOCS.split(",") if p.strip()]


def _build_state() -> dict:
    """Lazy one-time build. RAG mode = RETRIEVE-THEN-ANSWER over a CORPUS: one
    deterministic hybrid+rerank retrieval across all loaded reports, then ONE model call
    over those passages. This is reliable, unlike a free agentic tool-loop, which gpt-oss
    over-uses — it keeps calling search_documents on answerable questions until the cap
    cuts it off, producing a fallback even when the right passage was retrieved at rank 1.
    Falls back to a base agent if RAG is unavailable."""
    paths = [p for p in _resolve_doc_paths() if os.path.exists(p)]
    if paths:
        try:
            from framework.ingest import ingest
            from framework.rag.corpus import CorpusRetriever
            from framework.core.models import resolve_model
            docs = []
            for i, p in enumerate(paths, 1):          # Phase-6 cache: unchanged files are no-ops
                print(f"[serve] ingesting {i}/{len(paths)}: {os.path.basename(p)} ...", flush=True)
                docs.append(ingest(p))
            scorer = None
            if os.environ.get("RAG_RERANK") == "1":
                from framework.rag.rerank import default_scorer
                scorer = default_scorer()
            print(f"[serve] embedding + building hybrid index over {len(docs)} doc(s) "
                  "(first run per doc is slow on CPU; cached after) ...", flush=True)
            retriever = CorpusRetriever(docs, scorer=scorer)
            labels = ", ".join(retriever.doc_label(d.doc_id) for d in docs)
            print(f"[serve] RAG ready (retrieve-then-answer) over {len(docs)} doc(s): "
                  f"{labels} (rerank={'on' if scorer else 'off'}).", flush=True)
            return {"mode": "rag", "retriever": retriever,
                    "model": resolve_model(FRAMEWORK_MODEL), "docs": docs}
        except Exception as e:                        # never let RAG init crash the server
            print(f"[serve] RAG unavailable ({e}); base agent only.", flush=True)
    agent = Agent(FRAMEWORK_MODEL, tools=[get_weather, calculate],
                  system_prompt="You are Nova. Use tools when helpful. Be concise.",
                  memory=checkpoint("memory"),
                  middleware=[*resilience(), *guardrails(model_call_limit=MODEL_CALL_LIMIT)])
    return {"mode": "agent", "agent": agent}


def get_state() -> dict:
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def get_agent():
    """Back-compat entry point (serve.py warms via this) — builds the state once."""
    return get_state()


def _glossary_note(question: str, glossary: dict) -> str:
    """Prepend definitions for any abbreviations in the question, so acronym queries
    always carry the expansion regardless of what retrieval surfaced."""
    import re
    notes = [f"{t} = {glossary[t].expansion}" for t in set(re.findall(r"\b[A-Z]{2,8}\b", question))
             if t in glossary and glossary[t].expansion]
    return ("GLOSSARY: " + "; ".join(notes) + "\n\n") if notes else ""


def _context(question: str, st: dict) -> str:
    r = st["retriever"]
    hits = r._search(question, RAG_K)
    if not hits:
        return ""
    body = "\n\n".join(
        f"{r.doc_label(h['doc_id'])} — {' > '.join(h['section_path']) or 'doc'}, p{h['page']}:"
        f"\n{h['text']}" for h in hits)
    return _glossary_note(question, r.glossary) + body


def _prompt(question: str, ctx: str) -> str:
    return f"{RAG_SYS}\n\nPASSAGES:\n{ctx}\n\nQUESTION: {question}"


def rag_answer(question: str) -> str:
    st = get_state()
    if st["mode"] != "rag":
        return st["agent"].run(question)
    ctx = _context(question, st)
    if not ctx:
        return NOT_COVERED
    from langchain_core.messages import HumanMessage
    return st["model"].invoke([HumanMessage(content=_prompt(question, ctx))]).content


def rag_stream(question: str):
    st = get_state()
    if st["mode"] != "rag":
        yield from st["agent"].stream(question)
        return
    ctx = _context(question, st)
    if not ctx:
        yield NOT_COVERED
        return
    from langchain_core.messages import HumanMessage
    for chunk in st["model"].stream([HumanMessage(content=_prompt(question, ctx))]):
        text = getattr(chunk, "content", "") or ""
        if text:
            yield text


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
    try:
        return JSONResponse({"status": "ok", "answer": rag_answer(message)})
    except ApprovalRequired as a:                    # only the base-agent fallback can raise this
        return JSONResponse({"status": "approval_required",
                             "requests": a.requests, "thread_id": a.thread_id})


async def chat_stream(request):
    body = await _json(request)
    if body is None or not body.get("message"):
        return JSONResponse({"error": "missing 'message'"}, status_code=400)
    message = body["message"]

    def sse():
        try:
            for token in rag_stream(message):
                yield f"data: {token}\n\n"
        except ApprovalRequired as a:                # only the base-agent fallback can raise this
            yield f"data: [approval required for {a.requests}]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


async def approve(request):
    body = await _json(request)
    if body is None or "thread_id" not in body:
        return JSONResponse({"error": "missing 'thread_id'"}, status_code=400)
    st = get_state()
    if st["mode"] != "agent":                        # retrieve-then-answer has no approval flow
        return JSONResponse({"error": "no approval pending"}, status_code=400)
    try:
        answer = st["agent"].resume(body["thread_id"], approve=bool(body.get("approve")))
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
