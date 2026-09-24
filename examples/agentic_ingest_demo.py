"""Phase 5 Done-when: an agent NAVIGATES a real parsed document with the five
structure-aware tools (not one-shot search). Multi-hop question forces >1 tool call.

Run: uv run python examples/agentic_ingest_demo.py   (needs GROQ_API_KEY; bge-m3 +
reranker are local, no key.)
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))

from framework.ingest import ingest
from framework.rag.retriever import DocumentRetriever
from framework.rag.rerank import default_scorer
from framework.core.agent import Agent
from framework.middleware.guardrails import guardrails

NAV_TOOLS = {"search_documents", "list_outline", "get_section", "lookup_term", "get_table"}
SYSTEM = (
    "You answer ONLY from the knowledge base, via the tools. Orient with list_outline, "
    "resolve abbreviations with lookup_term, find facts with search_documents, read tables "
    "with get_table. CHAIN tools as needed and do not answer until every part is resolved."
)
QUESTION = ("What does the abbreviation PECA stand for, and according to the report how many "
            "prisoners were on death row in 2024?")


def tool_calls(events):
    return [e["tool"] for e in events if e["kind"] == "tool_call" and e.get("tool") in NAV_TOOLS]


def main():
    print("ingesting + indexing SHR (hybrid + reranker)...")
    doc = ingest("docs/SHR/2024_SHR_English.pdf", write=False)
    dr = DocumentRetriever(doc, scorer=default_scorer(), k=5)

    agent = Agent("groq:openai/gpt-oss-120b", tools=dr.as_tools(), system_prompt=SYSTEM,
                  middleware=guardrails(model_call_limit=8))
    res = agent.run_traced(QUESTION)
    calls = tool_calls(res["events"])
    print(f"\nQ: {QUESTION}")
    print(f"Answer: {res['answer']}")
    print(f"Tool calls ({len(calls)}): {calls}")
    print(f"DONE-WHEN >1 retrieval/nav call: {'PASS' if len(calls) > 1 else 'FAIL'}")


if __name__ == "__main__":
    main()
