"""Framework demo — see Phases 1-3 working end to end.

    uv run python examples/demo.py

Shows: tool use with a trajectory trace, multi-turn memory, the human-in-the-loop
safety gate (approve/deny), and structured output. Requires GROQ_API_KEY in .env.
"""

from dotenv import load_dotenv

from framework import Agent
from framework.memory import checkpoint
from framework.middleware import guardrails, ApprovalRequired
from framework.structured import extract
from framework.tools import get_weather, calculate, delete_record

load_dotenv()

MODEL = "groq:llama-3.3-70b-versatile"


def rule(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main():
    # 1) Tool use + observability
    rule("1. Tool use with a trajectory trace")
    agent = Agent(MODEL, tools=[get_weather, calculate], system_prompt="Use tools. Be concise.")
    r = agent.run_traced("What's the weather in Lahore, and what is 15% of 240?")
    print("Answer :", r["answer"])
    print("Trace  :", r["summary"])

    # 2) Multi-turn memory (per-session via thread_id)
    rule("2. Multi-turn memory")
    chat = Agent(MODEL, memory=checkpoint("memory"), system_prompt="Answer briefly.")
    chat.run("My favorite number is 42. Remember it.", thread_id="s1")
    print("Recall :", chat.run("What is my favorite number?", thread_id="s1"))

    # 3) Human-in-the-loop safety gate
    rule("3. Safety gate — destructive tool needs approval")
    guarded = Agent(
        MODEL,
        tools=[delete_record],
        system_prompt="Use delete_record when asked to delete.",
        memory=checkpoint("memory"),
        middleware=guardrails(require_approval=["delete_record"], model_call_limit=6),
    )
    try:
        guarded.run("Delete record 42.", thread_id="d1")
    except ApprovalRequired as a:
        print("Pending:", a.requests)
        print("Denying -> the delete never executes.")
        try:
            guarded.resume("d1", approve=False)
        except ApprovalRequired:
            pass  # model may re-request; still blocked

    # 4) Structured output (validated Pydantic)
    rule("4. Structured output")
    from pydantic import BaseModel

    class Ticket(BaseModel):
        summary: str
        priority: str
        category: str

    t = extract(MODEL, "Login returns 500 for all users since this morning. Urgent.", Ticket,
                instruction="Turn this into a support ticket.")
    print("Ticket :", t)


if __name__ == "__main__":
    main()
