# PROGRESS.md — Living Progress Tracker

> **Where you stand.** Claude Code updates this after every session. Glance here to see your
> current milestone and how each area scores on the three axes.
>
> Axes: 🧠 Concept (can explain) · 🔨 Implementation (built & works) · 🎯 Mastery (extend/debug unaided)
> Status: ✅ complete (all three green) · 🟡 in progress · ⬜ not started

---

## Current position

- **Current milestone:** 🎉 ALL 11 COMPLETE — program finished.
- **Theory level:** strong across the map. Both originally-weak spots (MCP, evaluation) cemented
  through building.
- **Implementation level:** M0–M11 all complete. Went from one raw LLM call to a deployable,
  production-lite agentic system (safety + resilience + observability + HTTP API).
- **Possible next directions:** real tools (replace stubs), swap in-memory vector store for a real
  vector DB, add prompt caching for real providers, containerize/deploy the API, expand the eval set.
- **Housekeeping:** two cleaned branches (HRCP-SHR-RAG, class-based-AI-Agent) still not pushed to
  GitHub — class-based-AI-Agent needs a force-push (history rewritten to purge docs/SHR).

---

## Milestone status

| # | Milestone | 🧠 | 🔨 | 🎯 | Status |
|---|-----------|----|----|----|--------|
| 0 | Single LLM call | 🟢 | 🟢 | 🟢 | ✅ |
| 1 | Multi-turn chatbot with memory | 🟢 | 🟢 | 🟢 | ✅ |
| 2 | System prompts & structured output | 🟢 | 🟢 | 🟢 | ✅ |
| 3 | Streaming & provider abstraction | 🟢 | 🟢 | 🟢 | ✅ |
| 4 | First tool (function calling) | 🟢 | 🟢 | 🟢 | ✅ |
| 5 | Multi-tool agent + agentic loop | 🟢 | 🟢 | 🟢 | ✅ |
| 6 | RAG (embeddings + vector search) | 🟢 | 🟢 | 🟢 | ✅ |
| 7 | MCP integration | 🟢 | 🟢 | 🟢 | ✅ *(theory spot now cemented)* |
| 8 | Agent orchestration | 🟢 | 🟢 | 🟢 | ✅ |
| 9 | Evaluation harness | 🟢 | 🟢 | 🟢 | ✅ *(theory spot now cemented)* |
| 10 | Safety & guardrails | 🟢 | 🟢 | 🟢 | ✅ *(mastery after 1 correction)* |
| 11 | Production capstone | 🟢 | 🟢 | 🟢 | ✅ |

*🟢 = demonstrated · 🟡 = partial / to reinforce · ⬜ = not yet*

---

## Session log

*(append newest at the top — one entry per session)*

### Session 11 — 2026-07-28 — Milestone 11 complete 🎉 PROGRAM FINISHED
- Built production capstone integrating everything: `observability.py` (TrajectoryLogger — timed
  structured events + summary), `resilient_llm.py` (retry w/ exponential backoff + provider
  fallback), `production/agent.py` (ProductionAgent tying safety + resilience + logging),
  `api.py` (starlette HTTP service: POST /chat, GET /health, agent built once at startup).
- Demos: `capstone_demo.py` (full trace + staged fallback via FlakyLLM), API verified with
  starlette TestClient (health, 400 validation, real /chat with trace).
- Live: staged flaky primary → fell back to groq; retry logic ALSO salvaged a real transient
  groq tool_use_failed error; send_email/delete_record blocked; full timed trajectory logged.
- **Nailed mastery (after one nudge):** full operator trace of a request — cost grows per round
  as history accumulates (call #2 pricier than #1), failure at each network boundary handled by
  retry→fallback→AllProvidersFailed, safety via authorize_tool gate + spend cap, API-layer 400
  validation. Also spotted unprompted that the agent lacked the MCP directory tool (tool_calls=0).
- **Journey:** M0 (one raw Groq call, no memory/tools/loop) → M11 (deployable agentic system).
  All three axes green on all 11 milestones. Both flagged weak theory spots closed by building.

### Session 10 — 2026-07-28 — Milestone 10 complete
- Built safety layer: `guardrails.py` (SpendTracker capped loop, DESTRUCTIVE_TOOLS set,
  pluggable approver always_deny/console_approver), `safe_runner.py` (charges cap per round,
  routes tools through authorize_tool, fences untrusted content as <untrusted_document>).
  Added destructive stub tools send_email + delete_record. Demo: `safety_demo.py`.
- All 3 defenses demonstrated live: (1) HITL gate blocked send_email, (2) spend cap halted a
  10-round task at the limit, (3) indirect prompt injection in a poisoned doc IGNORED — agent
  returned the real fact, no destructive tool executed.
- **Mastery after correction:** initially attributed the safety guarantee to prompt-level
  defenses (system prompt + XML fencing + round cap). Corrected: those are probabilistic; the
  real guarantee is the code-level `authorize_tool` gate that prevents func(**args) from ever
  running on a denied destructive call. Model DECISION vs code EXECUTION are separate layers;
  safety lives in the execution layer. Re-check confirmed understanding locked in.
- **Next:** Milestone 11 — production capstone (final).

### Session 9 — 2026-07-28 — Milestone 9 complete (evaluation theory spot cemented)
- Built eval harness: `dataset.py` (eval set with per-case judging: contains/trajectory/judge),
  `runner.py` (instrumented agent returning answer + tool trajectory), `judge.py` (LLM-as-judge
  with rubric → structured JSON score), `eval_demo.py` (two-axis scorecard). Live: 5/5 passed;
  judge gave a genuine 4/5 critique on the open-ended case.
- Proved the trajectory catch: a simulated lucky guess (right answer, empty trajectory) —
  outcome-only grading PASSES it, outcome+trajectory FAILS it. That's the core M9 lesson.
- **Nailed:** mastery — articulated outcome-vs-trajectory (right answer via broken path = luck,
  not process; fails on next input), LLM-as-judge with rubric criteria (faithfulness, relevance,
  completeness, usefulness) + trajectory metrics (goal completion, step efficiency).
- Evaluation was the 2nd flagged weak theory spot — now genuinely owned. Both weak spots closed.
- **Next:** Milestone 10 — safety & guardrails.

### Session 8 — 2026-07-28 — Milestone 8 complete
- Built orchestrator-worker pattern: `SubAgent` (role + system prompt, stateless/isolated),
  `Orchestrator` (plan via JSON → delegate in sequence passing context → synthesize).
  Team: researcher, writer, critic. Demo: `orchestrate_demo.py`.
- Live teaching moment: on the test task the researcher misread "iteration caps" (drifted from
  agentic-loop caps to AI-safety/AGI), and writer+critic built on the error — a textbook
  demonstration of the error-propagation failure mode the learner predicted.
- **Nailed:** mastery check answered up front — multi-agent hurts via error propagation
  (early hallucination becomes downstream truth), cost multiplication; rule: known workflow →
  code it / single agent, reserve orchestration for open-ended non-deterministic tasks.
- **Next:** Milestone 9 — evaluation harness (2nd flagged weak theory spot).

### Session 7 — 2026-07-28 — Milestone 7 complete (MCP theory spot cemented)
- Built full MCP stack over stdio: server (`mcp_server/server.py`, FastMCP, 3 company-directory
  tools), client (`agent/mcp/client.py` — discover + openai_tools schema adapter + call), host
  (`mcp_demo.py` — async agentic loop using discovered tools).
- Demonstrated runtime tool discovery + protocol-based execution. Contrast with M4/M5 hardwired
  TOOL_REGISTRY made the "standard vs custom integration" point land.
- Debugged: llama-3.1-8b-instant looped redundantly and hit the cap (good illustration of why
  caps matter) → switched demo to llama-3.3-70b-versatile + firmer stop instruction. Clean answers.
- **Nailed:** mastery check — strong explanation of client vs server vs config JSON roles.
  Correction: mcp_demo.py is the HOST not config; our build has no config JSON (wiring hardcoded
  in the `connect()` call) — a real client externalizes it to claude_desktop_config.json.
- MCP was one of two flagged weak theory spots — now genuinely owned.
- **Next:** Milestone 8 — agent orchestration.

### Session 6 — 2026-07-28 — Milestone 6 complete
- Built RAG pipeline: `chunker.py` (overlapping word chunks), `vector_store.py` (Google
  `gemini-embedding-001` embeddings + in-memory cosine similarity search), `_ask_with_rag`
  in Chatbot (retrieve top-3 → inject as context → grounded answer with source citation).
- Sample doc: docs/ai_agents.txt. Tested: correctly answered "What is the ReAct pattern?"
  from retrieved chunks with citation.
- Debugged: old embedding model `text-embedding-004` deprecated → switched to
  `gemini-embedding-001` (found via models.list()).
- **Nailed:** mastery check — production-grade answer on fixed-K failure modes (knowledge
  cutoff vs chunk poisoning) with fixes: parent-child retrieval, agentic multi-hop, two-stage
  retrieval + rerankers (bge/cohere), corrective RAG. Exceeded the bar.
- Side task: purged large `docs/SHR/` PDFs (128MB/110MB, over GitHub's 100MB limit) from
  both HRCP-SHR-RAG and class-based-AI-Agent history via git-filter-repo. Backup tags created.
  class-based-AI-Agent now needs a force-push (history rewritten); not yet pushed.
- **Next:** Milestone 7 — MCP integration.

### Session 5 — 2026-07-23 — Milestones 4 & 5 complete
- M4: Built tool calling — `send_with_tools` across all 5 providers, tool execution loop
  in Chatbot. Concept: "untrusted decision engine vs secure runtime."
- M5: Expanded to 5 tools (get_weather, calculate, unit_convert, compare_cities, get_time),
  ReAct system prompt for step-by-step reasoning, model-driven tool chaining across rounds.
  Tested: model chained 4 tools in 2 rounds to answer a multi-step question.
- **Nailed:** M5 mastery — distinguished workflow (M4, code-driven) vs loop (M5, LLM-driven),
  identified cost risk from autonomous tool selection + growing message history per round.
- **Shaky:** M4 round-trip trace was directionally correct but initially blurred two API calls
  into one (corrected).
- **Next:** Milestone 6 — RAG.

### Session 4 — 2026-07-23 — Milestone 4 complete
- Built tool calling: `TOOL_DEFINITIONS` (JSON schema), `TOOL_REGISTRY` (name→function map),
  `send_with_tools` on all 5 providers, `_ask_with_tools` loop in Chatbot with MAX_TOOL_ROUNDS=5.
- Two tools: `get_weather` (stub) and `calculate` (safe eval with character whitelist).
- Adapter differences: Groq/OpenAI/OpenRouter share `tool_calls` format, Anthropic uses
  `tool_use` blocks with `_convert_tool`, Google uses `function_call` with `FunctionDeclaration`.
- **Nailed:** concept — "untrusted decision engine vs secure runtime" framing, security
  implications of letting LLM execute directly.
- **Shaky:** mastery check round-trip trace was directionally correct but blurred the two
  separate API calls into one flow. Corrected: model returns tool_call object (API call 1),
  your code executes, sends result back as a message (API call 2), model forms final answer.
- **Next:** Milestone 5 — multi-tool agent + agentic loop.

### Session 3 — 2026-07-21 — Milestone 3 complete
- Built streaming support (`stream=True` flag, `_stream_and_collect` for typewriter UX).
- Built 5-provider abstraction: Groq, OpenRouter, Anthropic, Google, OpenAI — all behind
  `BaseLLM` contract with `send()` + `stream()`. Factory picks provider from `LLM_PROVIDER` env var.
- Key adapter differences handled: Anthropic splits system prompt into separate param,
  Google uses `user`/`model` roles and `generate_content` API, OpenRouter reuses OpenAI SDK
  with custom `base_url`.
- Debugged OpenRouter (wrong model ID) and Anthropic (no credits) — code was correct.
- **Nailed:** mastery check — explained 3-layer streaming pipeline (SSE → backend proxy → UI fetch)
  and both abstraction approaches (AI gateway vs custom adapter).
- **Next:** Milestone 4 — first tool (function calling).

### Session 2 — 2026-07-21 — Milestone 2 complete
- Built persona system prompt (Nova) with identity, behavior rules, honesty and security
  guardrails. Added structured output via `analyze()` method — one-shot JSON extraction
  with schema-focused system prompt and markdown-fence fallback parsing.
- **Nailed:** mastery check — traced the full robustness stack (prompt-level → Pydantic
  validation → grammar masking at inference) and remembered reasoning key for latency.
- **Next:** Milestone 3 — streaming & provider abstraction.

### Session 1 — 2026-07-20 — Milestone 1 complete
- Built multi-turn chatbot with proper memory: user messages appended before the call,
  assistant replies appended after. System prompt separated into `role: system`.
- Fixed: original code was injecting a hardcoded assistant message before calling the model
  (putting words in its mouth) and had a parameter name mismatch crash.
- **Nailed:** mastery check — identified context window overflow + cost scaling, gave 5
  strategies (sliding window, summarization, semantic search, knowledge graph, hybrid) with
  real tradeoffs. Noted "lost in the middle" research finding unprompted.
- **Shaky:** semantic search con was slightly off (described staleness as a vector DB flaw
  rather than a retrieval ranking problem — minor).
- M0 mastery also confirmed: token/billing understanding demonstrated in the memory discussion.
- **Next:** Milestone 2 — system prompts & structured output.

### Session 0 — system setup
- Established the learning system (CLAUDE.md engine, ROADMAP, this tracker).
- Assessed starting point: strong theory (~85% of design-conversation readiness), thin
  implementation (M0 only). Weakest theory spots: MCP and evaluation.
- **Nailed:** LLM fundamentals, memory-as-illusion, RAG, tools, loops, orchestration, safety.
- **Shaky:** MCP is a *protocol/standard* (not "JSON code"); evaluation needs structured method.
- **Next:** Milestone 1 — build the memory chat loop.
