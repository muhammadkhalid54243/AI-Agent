"""Milestone 8 demo: orchestrator-worker pattern.

An orchestrator plans a task, delegates steps to specialist sub-agents
(researcher -> writer -> critic), and synthesizes their outputs into one answer.

Run:
    python orchestrate_demo.py
"""

import os
import sys

from dotenv import load_dotenv

from agent.llms.factory import get_llm
from agent.orchestration.orchestrator import Orchestrator
from agent.orchestration.sub_agent import default_team

load_dotenv()


def main():
    provider = os.environ.get("LLM_PROVIDER", "groq")
    if provider == "groq" and not os.environ.get("GROQ_MODEL"):
        os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
    llm = get_llm(provider)

    team = default_team(llm)
    orchestrator = Orchestrator(llm, team)

    task = sys.argv[1] if len(sys.argv) > 1 else (
        "Write a short, punchy explainer on why AI agents need iteration caps."
    )

    print(f"TASK: {task}\n")
    result = orchestrator.run(task)
    print("=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
