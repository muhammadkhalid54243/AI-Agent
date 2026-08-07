from framework.eval.runner import score_case


class FakeAgent:
    def __init__(self, answer, events):
        self._a, self._e = answer, events

    def run_traced(self, message, thread_id=None):
        return {"answer": self._a, "summary": {}, "events": self._e}


def test_score_case_trajectory_pass():
    agent = FakeAgent("The result is 144.", [{"kind": "tool_call", "tool": "calculate"}])
    case = {"id": "m", "type": "trajectory", "input": "12*12?",
            "expect_contains": ["144"], "expect_tools": ["calculate"]}
    r = score_case(agent, None, case)
    assert r["passed"] is True


def test_score_case_trajectory_fails_on_missing_tool():
    agent = FakeAgent("The result is 144.", [])  # right answer, no tool call
    case = {"id": "m", "type": "trajectory", "input": "12*12?",
            "expect_contains": ["144"], "expect_tools": ["calculate"]}
    r = score_case(agent, None, case)
    assert r["passed"] is False  # trajectory check catches the lucky guess


import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_run_suite_mostly_passes():
    from framework import Agent
    from framework.middleware import resilience
    from framework.tools import get_weather, calculate
    from framework.eval import run_suite

    agent = Agent("groq:llama-3.3-70b-versatile", tools=[get_weather, calculate],
                  system_prompt="Use tools to answer accurately, then reply concisely.",
                  middleware=resilience())
    results = run_suite(agent)
    passed = sum(r["passed"] for r in results)
    assert passed >= 2  # at least the two trajectory cases
