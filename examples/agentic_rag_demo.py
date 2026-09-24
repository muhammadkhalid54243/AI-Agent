"""Phase 7 Done-when: prove the agent LOOPS over retrieval on a multi-hop question.

The corpus hides a THREE-hop bridge, deliberately built so the answer chunk shares
NO vocabulary with the question:
    Q: "In what year was the employee record of Project Helios's engineering lead opened?"
    hop 1  Helios   -> its lead's codename is "Falcon"          (idx0)
    hop 2  Falcon   -> Dr. Elena Ruiz, employee record E-1042   (idx2)
    hop 3  E-1042   -> record opened in 2019                    (idx4)

The year lives in idx4, keyed ONLY by the record id "E-1042" — it contains none of
the query's terms (no "Helios", "lead", "record ... opened" is generic). So a one-shot
semantic search CANNOT retrieve it: you only know to look for "E-1042" AFTER reading
idx2, which you only find AFTER reading idx0. Each hop is unlocked by the previous
one's content — the definition of a chain a single call cannot make. lookup_term
(exact id match) is what carries hops 2 and 3.

Run: uv run python examples/agentic_rag_demo.py
Needs GROQ_API_KEY (gpt-oss) + GOOGLE_API_KEY (embeddings).
"""

import sys

# Windows consoles choke on some model output (narrow no-break space, etc.).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from framework.core.agent import Agent
from framework.middleware.guardrails import guardrails
from framework.rag.retriever import Retriever

load_dotenv()

CORPUS = [
    "Project Helios is the company's satellite-imaging initiative. Its engineering "
    "lead is known internally only by the codename Falcon.",
    "The cafeteria menu rotates weekly; Fridays feature seafood paella.",
    "Codename Falcon refers to Dr. Elena Ruiz, whose personnel file is employee "
    "record E-1042.",
    "Parking permits are issued quarterly by the facilities team on level B2.",
    "Employee record E-1042 was opened in 2019, following an internal transfer.",
]

QUESTION = ("In what year was the employee record of Project Helios's engineering "
            "lead opened?")

SYSTEM = (
    "You answer ONLY from the knowledge base, never from prior knowledge. "
    "Use the retrieval tools and CHAIN them: if an answer depends on a fact you "
    "must first look up (a name, codename, or id), retrieve that first, read it, "
    "then issue a follow-up query for the next hop. Prefer lookup_term for exact "
    "names/codenames. Do not answer until the chain is fully resolved."
)


def retrieval_calls(events):
    tools = {"search_documents", "lookup_term", "get_section"}
    return [e["tool"] for e in events
            if e["kind"] == "tool_call" and e.get("tool") in tools]


def main():
    retriever = Retriever(CORPUS, k=2, chunk_size=300, overlap=20)

    print("=" * 70)
    print("AGENTIC (as_tools) — the loop")
    print("=" * 70)
    agent = Agent(
        "groq:openai/gpt-oss-120b",
        tools=retriever.as_tools(),
        system_prompt=SYSTEM,
        middleware=guardrails(model_call_limit=8),  # capped loop (non-negotiable)
    )
    res = agent.run_traced(QUESTION)
    calls = retrieval_calls(res["events"])
    print(f"Q: {QUESTION}")
    print(f"Answer: {res['answer']}")
    print(f"Retrieval calls ({len(calls)}): {calls}")
    print(f"Model calls: {res['summary']['model_calls']}")
    print(f"DONE-WHEN >1 retrieval call: {'PASS' if len(calls) > 1 else 'FAIL'}")

    print()
    print("=" * 70)
    print("SINGLE-TOOL baseline (as_tool) — semantic search only")
    print("=" * 70)
    # Honest note: this still runs in the agent loop, so it CAN call search more than
    # once. The real difference isn't call count — it's the missing tools (lookup_term
    # for exact-id hops, get_section for beyond-k). On a tiny corpus whose answer chunk
    # shares words with the question, semantic search alone may cover the hops and pass;
    # that edge shrinks as hops, distractors, and vocabulary distance grow.
    baseline = Agent(
        "groq:openai/gpt-oss-120b",
        tools=[retriever.as_tool()],
        system_prompt="Answer ONLY from the knowledge base using the search tool.",
        middleware=guardrails(model_call_limit=4),
    )
    res2 = baseline.run_traced(QUESTION)
    calls2 = retrieval_calls(res2["events"])
    print(f"Answer: {res2['answer']}")
    print(f"Retrieval calls ({len(calls2)}): {calls2}  (only search_documents available)")


if __name__ == "__main__":
    main()
