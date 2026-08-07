from framework.observability import TrajectoryTracer


def test_tracer_counts_model_and_tool_calls():
    t = TrajectoryTracer()
    t.on_chat_model_start({}, [], run_id="a")
    t.on_llm_end(None, run_id="a")
    t.on_tool_start({}, "", run_id="b", name="get_weather")
    t.on_tool_end("", run_id="b")
    s = t.summary()
    assert s["model_calls"] == 1
    assert s["tool_calls"] == 1
    assert s["total_ms"] >= 0
    assert any(e.get("tool") == "get_weather" for e in t.events)


import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_run_traced_reports_trajectory():
    from framework import Agent
    from framework.middleware import resilience
    from framework.tools import get_weather

    agent = Agent("groq:llama-3.3-70b-versatile", tools=[get_weather],
                  system_prompt="Use tools to answer.",
                  middleware=resilience())  # retry transient provider errors
    result = agent.run_traced("What is the weather in Lahore?")
    assert "42" in result["answer"]
    assert result["summary"]["model_calls"] >= 1
    assert result["summary"]["tool_calls"] >= 1
