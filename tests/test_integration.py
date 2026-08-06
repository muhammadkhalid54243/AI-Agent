import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # pick up GROQ_API_KEY from .env before the skip check

from framework import Agent
from framework.tools import get_weather, calculate

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY for a live run"
)


def test_live_tool_use_answers_weather():
    agent = Agent(
        "groq:llama-3.3-70b-versatile",
        tools=[get_weather, calculate],
        system_prompt="Use tools to answer accurately. Then reply in one sentence.",
    )
    out = agent.run("What is the temperature in Lahore?")
    assert "42" in out
