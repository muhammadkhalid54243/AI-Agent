from framework.orchestration import subagent


class FakeAgent:
    def run(self, question, thread_id=None):
        return f"handled: {question}"


def test_subagent_wraps_agent_as_named_tool():
    tool = subagent(FakeAgent(), "ask_researcher", "Delegate research questions.")
    assert tool.name == "ask_researcher"
    assert tool.invoke({"question": "what is RAG?"}) == "handled: what is RAG?"


import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_supervisor_delegates_to_specialists():
    from framework import Agent
    from framework.orchestration import subagent, supervisor
    from framework.middleware import resilience

    MODEL = "groq:llama-3.3-70b-versatile"
    researcher = Agent(MODEL, system_prompt="You list 2-3 concise factual bullet points.")
    writer = Agent(MODEL, system_prompt="You turn bullet points into one clear sentence.")

    orch = supervisor(MODEL, [
        subagent(researcher, "researcher", "Gather facts on a topic as bullet points."),
        subagent(writer, "writer", "Turn facts into prose."),
    ], middleware=resilience())

    out = orch.run("Explain what an API is, briefly.")
    assert isinstance(out, str) and len(out) > 0
