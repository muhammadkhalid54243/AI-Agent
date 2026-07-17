# PROGRESS.md — Living Progress Tracker

> **Where you stand.** Claude Code updates this after every session. Glance here to see your
> current milestone and how each area scores on the three axes.
>
> Axes: 🧠 Concept (can explain) · 🔨 Implementation (built & works) · 🎯 Mastery (extend/debug unaided)
> Status: ✅ complete (all three green) · 🟡 in progress · ⬜ not started

---

## Current position

- **Current milestone:** 1 — Multi-turn chatbot with memory
- **Theory level:** strong across the map (verified via concept tests). Reinforce **MCP** (M7)
  and **evaluation** (M9) while building them.
- **Implementation level:** early — one raw LLM call made (M0). Closing this gap is the mission.
- **Next action:** build the CLI chat loop that re-feeds `messages` history each turn.

---

## Milestone status

| # | Milestone | 🧠 | 🔨 | 🎯 | Status |
|---|-----------|----|----|----|--------|
| 0 | Single LLM call | 🟢 | 🟢 | 🟡 | 🟡 *(confirm token/billing explanation)* |
| 1 | Multi-turn chatbot with memory | 🟢 | ⬜ | ⬜ | ⬜ |
| 2 | System prompts & structured output | 🟢 | ⬜ | ⬜ | ⬜ |
| 3 | Streaming & provider abstraction | 🟡 | ⬜ | ⬜ | ⬜ |
| 4 | First tool (function calling) | 🟢 | ⬜ | ⬜ | ⬜ |
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

### Session 0 — system setup
- Established the learning system (CLAUDE.md engine, ROADMAP, this tracker).
- Assessed starting point: strong theory (~85% of design-conversation readiness), thin
  implementation (M0 only). Weakest theory spots: MCP and evaluation.
- **Nailed:** LLM fundamentals, memory-as-illusion, RAG, tools, loops, orchestration, safety.
- **Shaky:** MCP is a *protocol/standard* (not "JSON code"); evaluation needs structured method.
- **Next:** Milestone 1 — build the memory chat loop.
