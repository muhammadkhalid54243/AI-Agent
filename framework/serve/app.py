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
import time

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from framework import Agent
from framework.memory import checkpoint
from framework.middleware import resilience, ApprovalRequired
from framework.middleware.guardrails import guardrails
from framework.observability import metrics, tracing
from framework.rag.corpus import MIN_DENSE as _MIN_DENSE, MIN_RERANK as _MIN_RERANK
from framework.serve.history import History
from framework.structured.answer_parser import parse_units, render_units
from framework.tools import get_weather, calculate
from framework.serve.ui import INDEX_HTML

# Hard cap for the base-agent fallback path (non-RAG); the RAG path is a single
# model call, so it needs no loop cap. Non-negotiable per CLAUDE.md: loops are capped.
MODEL_CALL_LIMIT = int(os.environ.get("RAG_MODEL_CALL_LIMIT", "6"))

load_dotenv()


FRAMEWORK_MODEL = os.environ.get("FRAMEWORK_MODEL", "groq:openai/gpt-oss-120b")
# RAG_DOCS: a comma-separated list of DIRECTORIES and/or files — each dir expands to its
# *.pdf. Default is the full ENGLISH corpus (SHR annual + FFM + thematic); Urdu (docs/JH,
# docs/FFM-UR) is excluded by omission per the branch scope. RAG_DOC (single file) still wins.
RAG_DOCS = os.environ.get("RAG_DOCS", "docs/SHR,docs/FFM,docs/Thematic Reports")
RAG_K = int(os.environ.get("RAG_K", "8"))
# Deterministic abstention gate: below this retrieval confidence the server returns
# NOT_COVERED with NO model call. This moves the answer-vs-abstain decision out of the
# stochastic model (which flip-flopped on identical context) and into deterministic
# retrieval. Defaults are the calibrated constants in rag/corpus.py; override per deploy.
RAG_MIN_CONFIDENCE = float(os.environ.get("RAG_MIN_CONFIDENCE", _MIN_DENSE))
RAG_MIN_RERANK = float(os.environ.get("RAG_MIN_RERANK", _MIN_RERANK))
# Serve-time citation verification (Phase G): drop any fact whose cited quote isn't grounded
# in its passage. One extra model call per turn, so OFF by default (TPM/latency); the retrieval
# gate + grounded prompt already handle the common case. Turn on for maximum citation trust.
RAG_VERIFY = os.environ.get("RAG_VERIFY") == "1"
DEFAULT_THREAD = "default"
NOT_COVERED = ("The reports don't appear to cover that. Try a more specific fact, figure, "
               "or section.")
# Condense a follow-up into a standalone retrieval query (multi-turn). Deterministic, one call.
CONDENSE_SYS = (
    "Given the conversation so far and a follow-up question, rewrite the follow-up as a "
    "STANDALONE question that carries all the context it needs (resolve pronouns, 'that', "
    "years like 'what about 2023'). Output ONLY the rewritten question, nothing else.")
RAG_SYS = (
    "You answer questions about HRCP human-rights reports using ONLY the passages below. "
    "Each passage begins with its source line: Report — Section, pN.\n"
    "Rules:\n"
    "- Put each distinct fact on its OWN line, then its citation in brackets copied from that "
    "passage's source line — [Report — Section, pN], e.g. [SHR 2024 — Key issues, p18] — then, "
    "on the NEXT line, the exact sentence from the passage it rests on, in quotes, copied WORD "
    "FOR WORD (never paraphrased). Never cite a bare passage number like [3].\n"
    "- Leave a BLANK LINE between cited answers, so each fact + citation + quote is its own "
    "paragraph and the next one starts on a new line (not run onto the same line).\n"
    "- If the SAME statement appears in more than one report with the same wording (e.g. a "
    "declaration or law reproduced across years), give it ONCE and COMBINE the sources in a "
    "single citation — [SHR 2023; SHR 2024 — Section, pN]. Do NOT repeat the same text per report.\n"
    "- But when reports give DIFFERENT values for the same thing (a figure that changes by year), "
    "keep EACH on its own line with its own citation and quote — never merge differing numbers or "
    "silently pick just one year.\n"
    "- Use ONLY what the passages state; add nothing and never guess. If the passages do "
    "not contain a specific detail asked for, give what they DO state and omit the rest — "
    "the decision about whether the reports cover the question has already been made for "
    "you; your job is to report the passages faithfully.\n"
    "Be concise.")

_state = None
HISTORY = History()


def _resolve_doc_paths() -> list:
    """Which documents to serve. A single RAG_DOC overrides; otherwise RAG_DOCS is a
    comma-separated list of DIRECTORIES and/or files — each directory expands to its *.pdf.
    De-duped and sorted."""
    single = os.environ.get("RAG_DOC")
    if single:
        return [single]
    paths = []
    for entry in RAG_DOCS.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if os.path.isdir(entry):
            paths.extend(os.path.join(entry, f) for f in os.listdir(entry)
                         if f.lower().endswith(".pdf"))
        else:
            paths.append(entry)
    return sorted(dict.fromkeys(paths))     # de-dupe, stable order


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
                    "model": resolve_model(FRAMEWORK_MODEL, temperature=0),  # deterministic answers
                    "docs": docs}
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


def _page(h: dict) -> str:
    """Cite the PRINTED page when recovered; else fall back to the PDF index, marked
    explicitly (never print a wrong number silently)."""
    pp = h.get("printed_page")
    return f"p{pp}" if pp is not None else f"PDF p{h['page']}"


def _loc(h: dict) -> str:
    """Where a hit lives, for the citation. SHR docs carry an authoritative CHAPTER (from the TOC
    + vertical margin); use it. Other docs fall back to the font-heading section path."""
    ch = h.get("chapter")
    if ch:
        return ch
    return " > ".join(h.get("section_path") or []) or "doc"


def _passage_blocks(r, hits: list) -> list:
    """Each hit as a labelled passage block: 'Report — Chapter, pN:\\n<text>'."""
    return [f"{r.doc_label(h['doc_id'])} — {_loc(h)}, {_page(h)}:\n{h['text']}" for h in hits]


def _format_context(question: str, r, hits: list) -> str:
    """The labelled PASSAGES block for the prompt (empty if no hits)."""
    if not hits:
        return ""
    return _glossary_note(question, r.glossary) + "\n\n".join(_passage_blocks(r, hits))


def _context(question: str, st: dict) -> str:
    """Retrieve + format, in one call (kept for callers/tests that want the raw block)."""
    r = st["retriever"]
    return _format_context(question, r, r.retrieve(question, RAG_K))


def _answerable(r, hits: list) -> bool:
    """The deterministic abstention gate: answer only when retrieval is confident."""
    return bool(hits) and r.confident(
        hits, min_dense=RAG_MIN_CONFIDENCE, min_rerank=RAG_MIN_RERANK)


def _prompt(question: str, ctx: str, history: str = "") -> str:
    convo = f"CONVERSATION SO FAR:\n{history}\n\n" if history else ""
    return f"{RAG_SYS}\n\n{convo}PASSAGES:\n{ctx}\n\nQUESTION: {question}"


def _condense(model, history: str, question: str) -> str:
    """Rewrite a follow-up into a standalone retrieval query using the conversation. No-op
    (returns the question) when there's no history or on any failure — retrieval must never
    be blocked by the rewrite step."""
    if not history:
        return question
    from langchain_core.messages import HumanMessage
    try:
        msg = f"{CONDENSE_SYS}\n\nCONVERSATION:\n{history}\n\nFOLLOW-UP: {question}"
        out = (model.invoke([HumanMessage(content=msg)]).content or "").strip()
        return out or question
    except Exception:
        return question


def _claim_key(claim: str) -> str:
    """Normalized key for de-duping identical statements (case/space/trailing-punct-insensitive)."""
    import re
    return re.sub(r"\s+", " ", (claim or "")).strip().strip(".;:,").lower()


def _render_claims(claims: list) -> str:
    """Render verified claims through the SAME bullet renderer the output parser uses, so verified
    and unverified answers look identical: one bullet per cited answer, merged duplicate facts,
    quote on its own line."""
    units = [{"fact": c.get("claim", ""), "citation": c.get("cited_source", ""),
              "quote": c.get("supporting_quote", "")} for c in claims]
    return render_units(units)


def _authmap(r, hits: list) -> dict:
    """{(report_label, page_token): authoritative chapter} from the retrieved passages — the valid
    source lines the deterministic citation check holds every citation to. `_loc` derives the
    chapter from Part 1's TOC metadata, so this is the authority for 'what chapter is this page'."""
    return {(r.doc_label(h["doc_id"]), _page(h)): _loc(h) for h in hits}


def _verify_citations(answer: str, r, hits: list) -> str:
    """Deterministic citation check — the DEFAULT, zero model calls. Correct each citation's
    chapter against the passage it points to (or strip one no passage backs), and drop any quote
    that isn't verbatim in the passages. Degrade/correct ONLY — never drops a fact, so it can't
    regress coverage. Replaces the bare output parser on the served path, so every answer gets the
    same bullet shape AND a checked citation."""
    from framework.eval.cite_check import correct_citation
    from framework.eval.judge import quote_grounded
    units = parse_units(answer or "")
    if not units:
        return answer or ""                      # NOT_COVERED / unparseable -> passed through
    authmap = _authmap(r, hits)
    passages = _passage_blocks(r, hits)
    for u in units:
        u["citation"] = correct_citation(u.get("citation", ""), authmap)
        if u.get("quote") and not quote_grounded(u["quote"], passages):
            u["quote"] = ""                      # unverified quote removed; fact + page citation stay
    return render_units(units)


def _verify(st: dict, question: str, answer: str, passages: list, r, hits: list) -> str:
    """Model path (RAG_VERIFY=1): drop any fact the judge finds ungrounded or off-question, then
    apply the SAME deterministic citation correction to the survivors. Abstain if none survive."""
    from framework.eval.cite_check import correct_citation
    from framework.eval.judge import verify_answer
    try:
        claims = verify_answer(st["model"], question, answer, passages)
    except Exception:
        return answer                            # verification failure must not drop a good answer
    kept = [c for c in claims if c.get("grounded") and c.get("relevant", True)]
    if not kept:
        return NOT_COVERED
    authmap = _authmap(r, hits)
    for c in kept:
        c["cited_source"] = correct_citation(c.get("cited_source", ""), authmap)
    return _render_claims(kept)


def _trace_hits(hits: list) -> list:
    return [{"rank": i, "chunk_id": h.get("chunk_id"), "doc": h.get("doc_id"),
             "chapter": h.get("chapter"), "page": h.get("printed_page") or h.get("page")}
            for i, h in enumerate(hits[:RAG_K])]


def _used_chunk_ids(answer: str, r, hits: list) -> list:
    """Of the retrieved hits, which the final answer actually CITED — matched by the hit's report
    label AND its page token appearing in the answer's citations. Lets a trace show which of the
    retrieved chunks the answer drew on (vs retrieved-but-unused)."""
    import re
    a = answer or ""
    used = []
    for h in hits:
        pp = h.get("printed_page")
        pat = rf"\bp{pp}\b" if pp is not None else rf"PDF p{h.get('page')}\b"
        if h.get("chunk_id") and r.doc_label(h["doc_id"]) in a and re.search(pat, a):
            used.append(h["chunk_id"])
    return used


def _trace_usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None) or {}
    return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens")}


def rag_answer(question: str, thread_id: str = DEFAULT_THREAD) -> str:
    st = get_state()
    if st["mode"] != "rag":
        return st["agent"].run(question)
    r, model = st["retriever"], st["model"]
    t0 = time.perf_counter()
    with tracing.span("rag_answer", thread_id=thread_id, question=question):
        history = HISTORY.render(thread_id)
        with tracing.span("condense", has_history=bool(history)):
            tc = time.perf_counter()
            search_q = _condense(model, history, question)   # standalone query for retrieval
            tracing.set_attrs(condensed_query=search_q)
            if history:                                       # a model call happened only with history
                metrics.llm_call("condense"); metrics.latency("condense", time.perf_counter() - tc)
        with tracing.span("retrieve", k=RAG_K):
            tr0 = time.perf_counter()
            hits = r.retrieve(search_q, RAG_K)               # Phase D shape-dispatch; ONE retrieval
            metrics.latency("retrieve", time.perf_counter() - tr0)
            conf = r.confidence(hits) if (tracing.enabled() or metrics.enabled()) else None
            metrics.confidence(conf)
            if tracing.enabled():                            # compute trace attrs only when tracing
                tracing.set_attrs(n_hits=len(hits), confidence=conf, hits=_trace_hits(hits))
        answerable = _answerable(r, hits)
        tracing.set_attrs(abstained=not answerable, min_confidence=RAG_MIN_CONFIDENCE)
        metrics.request(abstained=not answerable)
        if not answerable:                                   # deterministic abstain — no answer call
            return NOT_COVERED
        passages = _passage_blocks(r, hits)
        ctx = _format_context(question, r, hits)
        from langchain_core.messages import HumanMessage
        with tracing.span("generate"):
            tg = time.perf_counter()
            resp = model.invoke([HumanMessage(content=_prompt(question, ctx, history))])
            answer = resp.content
            usage = _trace_usage(resp)
            tracing.set_attrs(**usage)
            metrics.llm_call("generate", usage.get("input_tokens"), usage.get("output_tokens"))
            metrics.latency("generate", time.perf_counter() - tg)
        if RAG_VERIFY:
            with tracing.span("verify"):
                metrics.llm_call("verify")                   # opt-in judge is one more model call
                answer = _verify(st, question, answer, passages, r, hits)
        else:
            with tracing.span("verify_citations"):       # deterministic default: 0 model calls
                answer = _verify_citations(answer, r, hits)
        if tracing.enabled():                            # which retrieved chunks the answer cited
            used = _used_chunk_ids(answer, r, hits)
            tracing.set_attrs(n_used=len(used), used_chunk_ids=used)
        tracing.set_attrs(answer=answer, answer_len=len(answer))
        metrics.latency("rag_answer", time.perf_counter() - t0)
        HISTORY.add(thread_id, question, answer)
        return answer


def rag_stream(question: str, thread_id: str = DEFAULT_THREAD):
    st = get_state()
    if st["mode"] != "rag":
        yield from st["agent"].stream(question)
        return
    if RAG_VERIFY:                                       # verify needs the whole answer; no token stream
        yield rag_answer(question, thread_id)
        return
    r, model = st["retriever"], st["model"]
    answer = NOT_COVERED
    t0 = time.perf_counter()
    # The whole span tree must CLOSE before we yield: this is a generator, and Starlette resumes it
    # in a different async context, so a span left open across the yield fails to detach. We buffer
    # the answer (the bullet parser needs it whole) and yield once, AFTER the span block.
    with tracing.span("rag_answer", thread_id=thread_id, question=question, streamed=True):
        history = HISTORY.render(thread_id)
        with tracing.span("condense", has_history=bool(history)):
            tc = time.perf_counter()
            search_q = _condense(model, history, question)
            tracing.set_attrs(condensed_query=search_q)
            if history:
                metrics.llm_call("condense"); metrics.latency("condense", time.perf_counter() - tc)
        with tracing.span("retrieve", k=RAG_K):
            tr0 = time.perf_counter()
            hits = r.retrieve(search_q, RAG_K)
            metrics.latency("retrieve", time.perf_counter() - tr0)
            conf = r.confidence(hits) if (tracing.enabled() or metrics.enabled()) else None
            metrics.confidence(conf)
            if tracing.enabled():
                tracing.set_attrs(n_hits=len(hits), confidence=conf, hits=_trace_hits(hits))
        answerable = _answerable(r, hits)
        tracing.set_attrs(abstained=not answerable, min_confidence=RAG_MIN_CONFIDENCE)
        metrics.request(abstained=not answerable)
        if answerable:
            ctx = _format_context(question, r, hits)
            from langchain_core.messages import HumanMessage
            with tracing.span("generate"):
                tg = time.perf_counter()
                parts = []
                for chunk in model.stream([HumanMessage(content=_prompt(question, ctx, history))]):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        parts.append(text)
                metrics.llm_call("generate"); metrics.latency("generate", time.perf_counter() - tg)
            with tracing.span("verify_citations"):       # deterministic default: 0 model calls
                answer = _verify_citations("".join(parts), r, hits)
            if tracing.enabled():                        # which retrieved chunks the answer cited
                used = _used_chunk_ids(answer, r, hits)
                tracing.set_attrs(n_used=len(used), used_chunk_ids=used)
            tracing.set_attrs(answer=answer, answer_len=len(answer))
            HISTORY.add(thread_id, question, answer)
        metrics.latency("rag_answer", time.perf_counter() - t0)
    yield answer


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
    thread_id = body.get("thread_id") or DEFAULT_THREAD
    try:
        return JSONResponse({"status": "ok", "answer": rag_answer(message, thread_id)})
    except ApprovalRequired as a:                    # only the base-agent fallback can raise this
        return JSONResponse({"status": "approval_required",
                             "requests": a.requests, "thread_id": a.thread_id})


async def chat_stream(request):
    body = await _json(request)
    if body is None or not body.get("message"):
        return JSONResponse({"error": "missing 'message'"}, status_code=400)
    message = body["message"]
    thread_id = body.get("thread_id") or DEFAULT_THREAD

    def sse():
        try:
            for token in rag_stream(message, thread_id):
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
