import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_agent_stream_yields_tokens():
    from framework import Agent
    agent = Agent("groq:llama-3.3-70b-versatile", system_prompt="Be concise.")
    tokens = list(agent.stream("Say hello in four words."))
    assert len(tokens) >= 1
    assert "".join(tokens).strip()


def test_health_and_ui_routes():
    from starlette.testclient import TestClient
    from framework.serve.app import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.get("/")
    assert r.status_code == 200 and "<form" in r.text


def test_chat_requires_message():
    from starlette.testclient import TestClient
    from framework.serve.app import app
    c = TestClient(app)
    assert c.post("/chat", json={}).status_code == 400
