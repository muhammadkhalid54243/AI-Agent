"""Milestone 11 — production capstone demo (CLI).

Runs the ProductionAgent end to end and prints the answer plus the full
structured trajectory (timing, tool calls, safety events). Also demonstrates
retry + provider fallback with a deliberately-failing primary provider.

Run:
    python capstone_demo.py
"""

import json
import os

from dotenv import load_dotenv

from agent.llms.factory import get_llm
from agent.safety.guardrails import Guardrails, always_deny
from agent.production.agent import ProductionAgent
from agent.production.resilient_llm import ResilientLLM
from agent.production.observability import TrajectoryLogger

load_dotenv()


class FlakyLLM:
    """A fake provider that always fails — used to prove retry + fallback work."""

    def send_with_tools(self, messages, tools):
        raise ConnectionError("simulated primary-provider outage")

    def send(self, messages):
        raise ConnectionError("simulated primary-provider outage")


def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    real_llm = get_llm(provider)

    # Primary is broken → resilient wrapper retries it, then falls back to the real one.
    shared_log = TrajectoryLogger("fallback-probe")
    resilient = ResilientLLM(
        providers=[("flaky-primary", FlakyLLM()), (provider, real_llm)],
        max_retries=1,
        base_delay=0.1,
        logger=shared_log,
    )

    guards = Guardrails(max_calls=8, approver=always_deny)
    agent = ProductionAgent(resilient, guards)

    print("=" * 68)
    print("CAPSTONE: production agent (resilience + safety + observability)")
    print("=" * 68)

    query = "Look up employee sara then delete her record."
    print(f"\nRequest: {query}\n")

    result = agent.run(query)

    print(f"Status:  {result['status']}")
    print(f"Answer:  {result['answer']}\n")
    print("Trace summary:")
    print(json.dumps(result["trace"], indent=2))

    print("\nProvider-fallback events (from the resilient wrapper):")
    for e in shared_log.events:
        print(f"  {e}")

    print("\nFull per-request event log:")
    for e in result["events"]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
