"""Compaction (mechanism 27) — summarise a long window, reinitialise from the summary.

Unlike clearing (a free, lossless mechanical edit), compaction is an LLM call and
is genuinely *lossy*: over-aggressive summarisation drops subtle context whose
importance only surfaces later. So the prompt is tuned for recall first, precision
second, and it explicitly preserves the load-bearing facts (decisions, constraints,
current goal, destructive actions already taken) while discarding the disposable
(redundant tool output, superseded reasoning, resolved sub-tasks).

Integrated path: `compaction()` wraps LangChain's SummarizationMiddleware, which
handles persistent history replacement (RemoveMessages) and keeps AI/Tool pairs
together. Standalone path: `compact()` summarises an explicit message list — used
by the recall probe and the future ContextManager seam.

Bounding: compaction only fires when context crosses the trigger fraction, and each
firing drops back under it, so the number of compactions is bounded by how often the
agent re-crosses the threshold — itself bounded by the agent's model-call cap. Run
without the middleware to disable entirely.
"""

from langchain_core.messages import SystemMessage

from framework.context.budget import measure_messages
from framework.core.models import resolve_model

COMPACTION_PROMPT = """You are compacting a long agent conversation. The text below \
will REPLACE the conversation history, so extract only what is needed to continue \
the task correctly. Favour recall: when unsure whether something matters, keep it.

PRESERVE (must appear if present):
- Architectural and design decisions, and the reasoning behind them.
- Established constraints and requirements.
- The current goal and the plan/next steps toward it.
- Unresolved problems still open.
- Destructive actions already taken (sends, deletes, writes) — the safety trail.
- ARTIFACTS written — notes/files created and WHAT THEY CONTAIN, by name (e.g. \
"saved the 24 order IDs to note ORDERS.md"). This pointer lets the agent re-read \
the artifact later; losing it strands data the agent already saved.

DISCARD:
- Redundant or verbatim tool output already acted on.
- Superseded intermediate reasoning and abandoned approaches (unless the fact that \
they were rejected matters).
- Resolved sub-tasks whose outcome is already captured above.

Respond ONLY with the compacted context, organised under: GOAL, DECISIONS, \
CONSTRAINTS, ARTIFACTS, OPEN PROBLEMS, ACTIONS TAKEN, NEXT STEPS.

<messages>
{messages}
</messages>"""


def _is_system(m) -> bool:
    if isinstance(m, dict):
        return m.get("role") == "system"
    return getattr(m, "type", "") == "system"


def _render(messages) -> str:
    lines = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "?")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def compact(messages, model, *, keep_recent=6, prompt=COMPACTION_PROMPT):
    """Summarise older messages into one system message, keeping system + recent verbatim.

    No-op (returns the same list) when there is nothing older than the recent window.
    """
    system = [m for m in messages if _is_system(m)][:1]
    body = [m for m in messages if not _is_system(m)]
    if len(body) <= keep_recent:
        return messages

    older, recent = body[:-keep_recent], body[-keep_recent:]
    llm = model if not isinstance(model, str) else resolve_model(model)
    summary = llm.invoke(prompt.format(messages=_render(older))).content
    summary_msg = SystemMessage(content="Summary of earlier conversation:\n" + summary)
    return system + [summary_msg] + recent


def _token_counter(messages) -> int:
    return measure_messages(list(messages))["total"]


def compaction(model, *, trigger_fraction=0.75, trigger_tokens=None, keep_messages=10):
    """Middleware factory — auto-summarise when context crosses the trigger.

    Default trigger is a fraction of the model's window (needs a model whose profile
    exposes max_input_tokens — real provider models do). Pass `trigger_tokens` for an
    absolute-token trigger instead (works with any model).

    Compose with clearing()/guardrails()/resilience(). Put clearing *before* this in a
    middleware list so we never pay to summarise payloads we were about to drop.
    """
    from langchain.agents.middleware import SummarizationMiddleware
    trigger = ("tokens", trigger_tokens) if trigger_tokens else ("fraction", trigger_fraction)
    return [SummarizationMiddleware(
        model=model,
        trigger=trigger,
        keep=("messages", keep_messages),
        token_counter=_token_counter,
        summary_prompt=COMPACTION_PROMPT,
    )]
