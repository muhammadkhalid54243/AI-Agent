# ROADMAP.md — Generative & Agentic AI Competency Map & Build Ladder

> **Where you want to be, and how to get there.**
> Two parts: (1) the *competency map* — the 9 skill dimensions that make up "can build
> agentic AI"; and (2) the *build ladder* — 11 milestones, each a small proof-of-concept you
> actually build. Every milestone strengthens specific competencies. Don't skip; each builds on
> the last.

---

## Part 1 — The competency map (9 dimensions)

These are the muscle groups. The build ladder trains them in a sensible order.

1. **Generative fundamentals** — prompting, system prompts, structured/JSON output, streaming,
   temperature & sampling, model/provider selection, cost & token awareness.
2. **Context & memory** — conversation state, context windows, history management, summarization,
   long-term memory files/stores.
3. **Retrieval / RAG** — embeddings, chunking, vector search, retrieval strategies, grounding
   answers, reducing hallucination.
4. **Tools & function calling** — defining tools, the request→execute→return cycle, multi-tool
   selection, MCP (the *protocol* that standardizes tool access — "USB-C for AI tools").
5. **Agentic patterns** — the think→act→check loop (ReAct), planning, reflection/self-critique,
   stop conditions.
6. **Orchestration** — multi-agent systems, orchestrator-worker (supervisor) pattern, routing,
   sub-agents, when *not* to use multiple agents.
7. **Evaluation** — eval sets, outcome vs. trajectory checks, LLM-as-judge, catching regressions.
8. **Safety & guardrails** — permission gates, sandboxing, spend/iteration caps, scoped tools,
   prompt-injection defense.
9. **Production** — streaming UX, prompt caching, latency, observability/logging, robust error
   handling, deployment.

---

## Part 2 — The build ladder (11 milestones)

Each milestone lists: **Concept** (the idea + analogy to anchor it), **Build** (the PoC),
**Competencies** (which dimensions it trains), and **Mastery check** (how you prove you own it,
not just that it runs).

---

### Milestone 0 — Single LLM call ✅ *(done)*
- **Concept:** one prompt in, one completion out. The atom of everything.
- **Build:** a script that sends one message and prints the reply. *(Done via Groq.)*
- **Competencies:** 1.
- **Mastery check:** explain what a token is and why you're billed per token. *(Revisit if unsure.)*

### Milestone 1 — Multi-turn chatbot with memory
- **Concept:** the model has *zero* memory — every call is amnesia. "Memory" is an illusion you
  engineer by re-feeding the conversation each turn. (The *Memento* notebook.)
- **Build:** a CLI chat loop that keeps a `messages` history list and re-sends it each turn, so
  the bot "remembers" earlier turns.
- **Competencies:** 1, 2.
- **Mastery check:** what happens as the history grows past the context window — and name one fix.

### Milestone 2 — System prompts & structured output
- **Concept:** the system prompt sets the model's role/rules (the job description). Structured
  output = forcing the model to answer in strict JSON so *code* can use it, not just humans.
- **Build:** give your bot a persona via a system prompt; then add a mode that returns valid JSON
  you parse and act on (e.g. `{ "intent": ..., "entities": [...] }`).
- **Competencies:** 1.
- **Mastery check:** the model returns slightly-malformed JSON — how do you make this robust?

### Milestone 3 — Streaming & provider abstraction
- **Concept:** streaming = tokens arrive as they're generated (the typewriter effect) instead of
  waiting for the whole reply. Abstraction = one function to call *any* provider, so swapping
  Claude↔Gemini↔Groq is a one-line change.
- **Build:** add token streaming to the CLI; wrap your call behind a single `chat(provider, messages)`
  function that routes to Anthropic / Google / Groq / OpenRouter.
- **Competencies:** 1, 9.
- **Mastery check:** add a new provider in under 5 minutes without touching the rest of the app.

### Milestone 4 — Your first tool (function calling)
- **Concept:** a tool lets the model *do*, not just *say*. Crucial truth: the model doesn't run
  the tool — it *requests* a call; **your code runs it** and hands the result back.
- **Build:** give the bot one tool (e.g. a calculator or a `get_weather()` stub). Implement the
  full request→execute→return-result→final-answer cycle.
- **Competencies:** 4.
- **Mastery check:** trace the exact round-trips between model and your code for one tool call.

### Milestone 5 — Multi-tool agent + the agentic loop (ReAct)
- **Concept:** the leap to *agent*. The model decides *which* tool, *when*, and *whether it's
  done*, looping think→act→observe→repeat until the goal is met. (The chef tasting the soup.)
- **Build:** 2–3 tools + a loop where the model chains tool calls to solve a multi-step task,
  with a **hard max-iteration cap** and a clear stop condition.
- **Competencies:** 4, 5, 8 (caps).
- **Mastery check:** show where an uncapped loop would burn money, and how your cap prevents it.

### Milestone 6 — RAG (embeddings + vector search)
- **Concept:** ground answers in *your* documents. Embeddings turn text into vectors (meaning as
  coordinates); vector search finds the chunks closest in *meaning*; you inject those into the
  prompt so the model answers from real sources, not vibes.
- **Build:** embed a small doc set, store vectors, retrieve top-k for a query, augment the prompt,
  answer with sources. (A local/in-memory vector store is fine to start.)
- **Competencies:** 3.
- **Mastery check:** why does semantic (vector) search beat keyword search — give a failing keyword case.

### Milestone 7 — MCP integration
- **Concept:** MCP (Model Context Protocol) is a *standard*, not code — a universal plug shape
  ("USB-C for AI tools"). Any MCP-compatible client can use any MCP server without custom wiring.
  *(This is one of your two theory spots to solidify — building it will cement it.)*
- **Build:** connect your agent to an existing MCP server (or write a tiny one) and let the agent
  use its tools through the protocol rather than hand-wired functions.
- **Competencies:** 4.
- **Mastery check:** explain the difference between an MCP *server*, *client*, and the config JSON.

### Milestone 8 — Agent orchestration (orchestrator + sub-agents)
- **Concept:** split a big job across specialized agents. An **orchestrator** delegates subtasks to
  **sub-agents** (researcher, coder, checker) and combines results. Know when *one* agent is better.
- **Build:** an orchestrator that routes a task to 2–3 specialized sub-agents and synthesizes their
  outputs.
- **Competencies:** 5, 6.
- **Mastery check:** give one task where multi-agent *hurts* vs. helps, and explain why.

### Milestone 9 — Evaluation harness
- **Concept:** stop judging by vibes. Build an eval set (tasks with known-good outcomes), check
  *outcome and trajectory* (did it use the right steps, not just luck into the answer), and use
  LLM-as-judge to scale grading. *(Your second theory spot — building this cements it.)*
- **Build:** a small eval set for your agent + a runner that scores pass/fail and flags regressions
  when you change a prompt or model.
- **Competencies:** 7.
- **Mastery check:** your agent gets the right answer via a broken path — why is that still a fail?

### Milestone 10 — Safety & guardrails
- **Concept:** an autonomous agent with tools can do real damage. Layer defenses: permission gates
  on risky actions, spend/iteration caps, scoped/least-privilege tools, sandboxing, and prompt-
  injection defense (never trust text the model *read* as if it were an instruction *you* gave).
- **Build:** add human-in-the-loop confirmation for write/send/delete tools; add a spend cap; add a
  simple prompt-injection test and defense.
- **Competencies:** 8.
- **Mastery check:** demonstrate a prompt-injection attempt against your agent and how you block it.

### Milestone 11 — Production capstone
- **Concept:** turn a PoC into something real: caching to cut cost/latency, observability so you can
  see what the agent did, graceful error handling, and deployment.
- **Build:** take one earlier agent to "production-lite" — prompt caching, structured logging of the
  agent's trajectory, retries/fallbacks, and a deployable interface (API endpoint or simple UI).
- **Competencies:** 1, 5, 7, 8, 9 (integration of everything).
- **Mastery check:** walk through a request end-to-end and point out every cost, failure, and safety
  control along the path.

---

## How to progress

- One milestone at a time, in order. Each is a small, finishable PoC — build it, then prove you own
  it via the mastery check before moving on.
- A milestone counts as **✅ complete** only when all three axes are green: 🧠 Concept, 🔨
  Implementation, 🎯 Mastery. See `PROGRESS.md` for your live status.
- It's fine to reuse/extend earlier PoCs rather than starting fresh each time — a growing codebase
  is a feature.

## Provider notes

- **Learning default:** cheap + fast models (Claude Haiku, Gemini Flash, Llama-8B on Groq).
- **Anthropic (Claude):** native SDK; best for tool-use, MCP, and the agentic milestones.
- **Groq:** OpenAI-compatible, very fast — great sandbox for chat/loop experiments.
- **Google AI Studio (Gemini):** generous free tier, huge context — good for RAG experiments.
- **OpenRouter:** one key, many models — handy for comparing behavior across models.
- Keep everything behind your Milestone-3 abstraction so switching providers is trivial.
