# PROGRESS.md — Living Progress Tracker

> **Where you stand.** Claude Code updates this after every session. Glance here to see your
> current milestone and how each area scores on the three axes.
>
> Axes: 🧠 Concept (can explain) · 🔨 Implementation (built & works) · 🎯 Mastery (extend/debug unaided)
> Status: ✅ complete (all three green) · 🟡 in progress · ⬜ not started

---

## Current position

- **Current milestone:** 5 — Multi-tool agent + agentic loop
- **Theory level:** strong across the map (verified via concept tests). Reinforce **MCP** (M7)
  and **evaluation** (M9) while building them.
- **Implementation level:** M0–M4 complete. Tool calling with 2 tools across 5 providers.
- **Next action:** Milestone 5 — multi-tool agent with ReAct loop, hard iteration cap, stop condition.

---

## Milestone status

| # | Milestone | 🧠 | 🔨 | 🎯 | Status |
|---|-----------|----|----|----|--------|
| 0 | Single LLM call | 🟢 | 🟢 | 🟢 | ✅ |
| 1 | Multi-turn chatbot with memory | 🟢 | 🟢 | 🟢 | ✅ |
| 2 | System prompts & structured output | 🟢 | 🟢 | 🟢 | ✅ |
| 3 | Streaming & provider abstraction | 🟢 | 🟢 | 🟢 | ✅ |
| 4 | First tool (function calling) | 🟢 | 🟢 | 🟢 | ✅ |
| 5 | Multi-tool agent + agentic loop | 🟢 | ⬜ | ⬜ | ⬜ |
| 6 | RAG (embeddings + vector search) | 🟢 | ⬜ | ⬜ | ⬜ |
| 7 | MCP integration | 🟡 | ⬜ | ⬜ | ⬜ *(theory to reinforce)* |
| 8 | Agent orchestration | 🟢 | ⬜ | ⬜ | ⬜ |
| 9 | Evaluation harness | 🟡 | ⬜ | ⬜ | ⬜ *(theory to reinforce)* |
| 10 | Safety & guardrails | 🟢 | ⬜ | ⬜ | ⬜ |
| 11 | Production capstone | 🟡 | ⬜ | ⬜ | ⬜ |

*🟢 = demonstrated · 🟡 = partial / to reinforce · ⬜ = not yet*

---

## Session log

*(append newest at the top — one entry per session)*

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
